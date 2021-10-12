import os
import numpy as np
import requests
from io import BytesIO
import base64
from math import trunc
from PIL import Image as PILImage
import json
from glob import glob


def display_image(image_path, annotations, show_polys=True, show_bbox=True, show_crowds=True):
    image_name = image_path.split('\\')[-1]
    print('Image:')
    print(f'{image_name}')

    image = PILImage.open(image_path)

    buffer = BytesIO()
    image.save(buffer, format='jpeg')
    buffer.seek(0)

    data_uri = base64.b64encode(buffer.read()).decode('ascii')
    image_path = "data:image/jpeg;base64,{0}".format(data_uri)

    # Calculate the size and adjusted display size
    max_width = 600
    image_width, image_height = image.size
    adjusted_width = min(image_width, max_width)
    adjusted_ratio = adjusted_width / image_width
    adjusted_height = adjusted_ratio * image_height

    # Create list of polygons to be drawn
    polygons = {}
    bbox_polygons = {}
    rle_regions = {}
    poly_colors = {}
    for i, segm in enumerate(annotations):
        polygons_list = []
        if segm.get('is_crowd', 0) != 0:
            # Gotta decode the RLE
            px = 0
            x, y = 0, 0
            rle_list = []
            for j, counts in enumerate(segm['segmentation']['counts']):
                if j % 2 == 0:
                    # Empty pixels
                    px += counts
                else:
                    # Need to draw on these pixels, since we are drawing in vector form,
                    # we need to draw horizontal lines on the image
                    x_start = trunc(trunc(px / image_height) * adjusted_ratio)
                    y_start = trunc(px % image_height * adjusted_ratio)
                    px += counts
                    x_end = trunc(trunc(px / image_height) * adjusted_ratio)
                    y_end = trunc(px % image_height * adjusted_ratio)
                    if x_end == x_start:
                        # This is only on one line
                        rle_list.append({'x': x_start, 'y': y_start, 'width': 1, 'height': (y_end - y_start)})
                    if x_end > x_start:
                        # This spans more than one line
                        # Insert top line first
                        rle_list.append(
                            {'x': x_start, 'y': y_start, 'width': 1, 'height': (image_height - y_start)})

                        # Insert middle lines if needed
                        lines_spanned = x_end - x_start + 1  # total number of lines spanned
                        full_lines_to_insert = lines_spanned - 2
                        if full_lines_to_insert > 0:
                            full_lines_to_insert = trunc(full_lines_to_insert * adjusted_ratio)
                            rle_list.append(
                                {'x': (x_start + 1), 'y': 0, 'width': full_lines_to_insert, 'height': image_height})

                        # Insert bottom line
                        rle_list.append({'x': x_end, 'y': 0, 'width': 1, 'height': y_end})
            if len(rle_list) > 0:
                rle_regions[segm['id']] = rle_list
        elif segm.get('segmentation', False):
            # Add the polygon segmentation
            for segmentation_points in segm['segmentation']:
                segmentation_points = np.multiply(segmentation_points, adjusted_ratio).astype(int)
                polygons_list.append(str(segmentation_points).lstrip('[').rstrip(']'))
        polygons[i] = polygons_list
        poly_colors[i] = 'blue'

        bbox = segm['bbox']
        bbox_points = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                       bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3],
                       bbox[0], bbox[1]]
        bbox_points = np.multiply(bbox_points, adjusted_ratio).astype(int)
        bbox_polygons[i] = str(bbox_points).lstrip('[').rstrip(']')

    # Draw segmentation polygons on image
    html = '<h3>Image: {}</h3>'.format(image_name)
    html += '<div class="container" style="position:relative;">'
    html += '<img src="{}" style="position:relative;top:0px;left:0px;width:{}px;">'.format(image_path,
                                                                                           adjusted_width)
    html += '<div class="svgclass"><svg width="{}" height="{}">'.format(adjusted_width, adjusted_height)

    poly_colors[0] = 'white'

    if show_polys:
        for seg_id, points_list in polygons.items():
            fill_color = poly_colors[seg_id]
            stroke_color = poly_colors[seg_id]
            for points in points_list:
                html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5" />'.format(
                    points, fill_color, stroke_color)

    if show_crowds:
        for seg_id, rect_list in rle_regions.items():
            fill_color = poly_colors[seg_id]
            stroke_color = poly_colors[seg_id]
            for rect_def in rect_list:
                x, y = rect_def['x'], rect_def['y']
                w, h = rect_def['width'], rect_def['height']
                html += '<rect x="{}" y="{}" width="{}" height="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5; stroke-opacity:0.5" />'.format(
                    x, y, w, h, fill_color, stroke_color)

    if show_bbox:
        for seg_id, points in bbox_polygons.items():
            fill_color = poly_colors[seg_id]
            stroke_color = poly_colors[seg_id]
            html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0" />'.format(
                points, fill_color, stroke_color)

    html += '</svg></div>'
    html += '</div>'
    html += '<style>'
    html += '.svgclass { position:absolute; top:0px; left:0px;}'
    html += '</style>'
    return html


if __name__ == '__main__':
    # view
    image_html = ""
    test_or_train = "test"
    annotation_path = rf'../datasets/luna16/prepared_data/annotations/instances_{test_or_train}.json'
    json_file = open(annotation_path)
    data = json.load(json_file)

    image_paths = glob(rf'../datasets/luna16/prepared_data/{test_or_train}/images/*.jpeg')
    # for path in image_paths:
    #     file_name = path.split('\\')[-1]
    #     ids = [img['id'] for img in data['images'] if img['file_name'] == file_name]
    #     annotations = [ann for ann in data['annotations'] if ann['image_id'] in ids]
    #     image_html += display_image(path, annotations, show_polys=False)
    #
    # with open(r'./image_display.html', 'w') as outfile:
    #     outfile.write(image_html)

    # view results
    image_html = ""
    test_or_train = "test"
    annotation_path = rf'../results/test_bbox_results_d0.json'
    json_file = open(annotation_path)
    result_data = json.load(json_file)

    image_paths = glob(rf'../datasets/luna16/prepared_data/test/images/*.jpeg')
    for path in image_paths:
        file_name = path.split('\\')[-1]
        ids = [img['id'] for img in data['images'] if img['file_name'] == file_name]
        annotations = [ann for ann in data['annotations'] if ann['image_id'] in ids]
        annotations += [ann for ann in result_data if ann['image_id'] in ids]
        image_html += display_image(path, annotations, show_polys=False)

    with open(r'./image_display_d0.html', 'w') as outfile:
        outfile.write(image_html)