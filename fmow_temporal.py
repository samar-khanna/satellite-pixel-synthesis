import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
import numpy as np
from shapely.wkt import loads as shape_loads

import tensor_transforms as tt
import pandas as pd


CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]


def is_invalid_lon_lat(lon, lat):
    return np.isnan(lon) or np.isnan(lat) or \
        (lon in [float('inf'), float('-inf')]) or (lat in [float('inf'), float('-inf')]) or \
        lon < -180 or lon > 180 or lat < -90 or lat > 90


def fmow_temporal_images(example, img_transform, num_frames=3, stack_tensor=True, channel_first=False):
    image_keys = sorted([k for k in example if k.endswith('.npy')])
    metadata_keys = sorted([k for k in example if k.endswith('.json')])
    if len(image_keys) < num_frames:
        while len(image_keys) < num_frames:
            image_keys.append('input-0.npy')
            metadata_keys.append('metadata-0.json')
    else:
        img_md = random.sample(list(zip(image_keys, metadata_keys)), k=num_frames)
        image_keys = [img for img, md in img_md]
        metadata_keys = [md for img, md in img_md]

    img = [img_transform(example[k]) for k in image_keys]
    if stack_tensor:
        img = torch.stack(img)  # (T, C, H, W)
        if channel_first:
            img = img.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

    return img, metadata_keys


def fmow_numerical_metadata(example, meta_df, target_resolution, num_metadata, rgb_key='input.npy',
                            md_key='metadata.json', base_year=1980, base_lon=180, base_lat=90):
    md = example[md_key]
    h, w, c = example[rgb_key].shape
    assert c == 3, 'Shape error'
    orig_res = min(h, w)

    target_res = target_resolution
    scale = orig_res / target_res
    gsd = md['gsd'] * scale

    cloud_cover = md['cloud_cover'] / 100.

    timestamp = md['timestamp']
    year = int(timestamp[:4]) - base_year
    month = int(timestamp[5:7])
    day = int(timestamp[8:10])
    hour = int(timestamp[11:13])

    # name_items = example['__key__'].split('-')[-1].replace('_rgb', '').split('_')
    name_items = example[md_key]['img_filename'].replace('_rgb', '').replace('.jpg', '').replace('_ms.tif', '').split('_')
    category = CATEGORIES[example['output.cls']] if 'output.cls' in example else example['category.txt']  # eg: recreational_facility
    location_id = int(name_items[-2])  # eg: 890
    image_id = int(name_items[-1])  # eg: 4

    polygon = meta_df[
        (meta_df['category'] == category) &
        (meta_df['location_id'] == location_id) &
        (meta_df['image_id'] == image_id)
    ]['polygon']
    assert len(polygon) == 1, f"{category}, {location_id}, {image_id} is not found in csv"
    poly = shape_loads(polygon.iloc[0])
    lon, lat = poly.centroid.x, poly.centroid.y
    assert not is_invalid_lon_lat(lon, lat)

    return torch.tensor([lon + base_lon, lat + base_lat, gsd, cloud_cover, year, month, day])


def fmow_temporal_preprocess_train(examples, img_transform, fmow_meta_df, resolution, num_cond=2):
    for example in examples:
        img_temporal, md_keys = fmow_temporal_images(example, img_transform, num_frames=num_cond + 1)

        target_img = img_temporal[0]  # (C, H, W)
        target_rgb_key = md_keys[0].replace('metadata', 'input').replace('json', 'npy')
        target_md = fmow_numerical_metadata(example, fmow_meta_df, resolution, 7, rgb_key=target_rgb_key, md_key=md_keys[0])
        year, month = target_md[-3], target_md[-2]
        t = year * 100 + month
        cond_t = tt.convert_to_coord_uneven_t(1, resolution, resolution, t, 5000)
        # target_md = metadata_normalize(target_md)

        # md_tensor = []
        # for md_key in md_keys[1:]:
        #     rgb_key = md_key.replace('metadata', 'input').replace('json', 'npy')
        #     md = fmow_numerical_metadata(example, fmow_meta_df, resolution, 7, rgb_key=rgb_key, md_key=md_key)
        #     year, month = md[-3], md[-2]
        #     md_tensor.append((year, month))

        target_img = torch.cat((target_img, cond_t), dim=0)  # (C+3, H, W)
        cond1 = img_temporal[1]
        cond2 = img_temporal[2]
        # output = {
        #     'target': target_img,
        #     'cond1': cond1,
        #     'cond2': cond2,
        # }
        yield target_img, cond1, cond2


def fmow_temporal_attpatch_preprocess_train(examples, img_transform, fmow_meta_df, resolution, crop_size, num_cond=2, is_test=False,):
    for example in examples:
        img_temporal, md_keys = fmow_temporal_images(example, img_transform, num_frames=num_cond + 1)

        target_img = img_temporal[0]  # (C, H, W)
        target_rgb_key = md_keys[0].replace('metadata', 'input').replace('json', 'npy')
        target_md = fmow_numerical_metadata(example, fmow_meta_df, resolution, 7, rgb_key=target_rgb_key, md_key=md_keys[0])
        year, month = target_md[-3], target_md[-2]
        t = year * 100 + month
        cond_t = tt.convert_to_coord_uneven_t(1, resolution, resolution, t, 5000)
        # target_md = metadata_normalize(target_md)

        # md_tensor = []
        # for md_key in md_keys[1:]:
        #     rgb_key = md_key.replace('metadata', 'input').replace('json', 'npy')
        #     md = fmow_numerical_metadata(example, fmow_meta_df, resolution, 7, rgb_key=rgb_key, md_key=md_key)
        #     year, month = md[-3], md[-2]
        #     md_tensor.append((year, month))

        target_img = torch.cat((target_img, cond_t), dim=0)  # (C+3, H, W)
        cond1 = img_temporal[1]
        cond2 = img_temporal[2]

        if is_test:
            data = {}
            for i in range(resolution // crop_size):
                for j in range(resolution // crop_size):
                    target_img = tt.patch_crop_dim3(target_img, i * crop_size, j * crop_size, crop_size)
                    cond1 = tt.patch_crop_dim3(cond1, i * crop_size, j * crop_size, crop_size)
                    cond2 = tt.patch_crop_dim3(cond2, i * crop_size, j * crop_size, crop_size)
                    data[(i, j)] = (target_img, cond1, cond2, i * crop_size, j * crop_size)
            yield data
        else:
            target_img, h_start, w_start = tt.random_crop_dim3(target_img, crop_size)
            cond1 = tt.patch_crop_dim3(cond1, h_start, w_start, crop_size)
            cond2 = tt.patch_crop_dim3(cond2, h_start, w_start, crop_size)
            yield target_img, cond1, cond2, h_start, w_start


def collate_fn(examples):
    target_imgs = []
    cond1_imgs = []
    cond2_imgs = []

    for example in examples:
        target_imgs.append(example['target'])
        cond1_imgs.append(example['cond1'])
        cond2_imgs.append(example['cond2'])

    return torch.stack(target_imgs), torch.stack(cond1_imgs), torch.stack(cond2_imgs)
