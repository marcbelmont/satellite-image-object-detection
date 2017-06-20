from PIL import Image
from collections import defaultdict
from os import makedirs
from params import IMG_SIZE
import pandas as pd

# Initial data:
#              id           image  class         detections
# 1  TQ2379_0_0_B  TQ2379_0_0.jpg      B  1776:520|1824:125
# 2  TQ2379_0_0_C  TQ2379_0_0.jpg      C           1760:456
# 3  TQ2379_0_0_D  TQ2379_0_0.jpg      D           1609:943
# 5  TQ2379_0_0_F  TQ2379_0_0.jpg      F          1937:1932
# 9  TQ2379_0_1_A  TQ2379_0_1.jpg      A           1940:204

# Vehicle instances (9):
# A 0   258    Motorcycle (12px)
# B 1   2629   Light short rear (30px)
# C 2   1035   Light long rear
# D 3   2659   Dark short rear
# E 4   825    Dark long rear
# F 5   539    Red short rear
# G 6   170    Red long rear
# H 7   471    Light van (40px)
# I 8   68     Red and white bus (45px)

SIZE = 2000
DIMS = dict(zip(range(9), [12, 30, 30, 30, 30, 30, 30, 40, 45]))


def extract_crops_sw(df, size, save_images, step):
    "Extract images using a sliding window. Save vehicles coordinates."
    # Encode class
    encoder = dict(zip('ABCDEFGHI', range(50)))
    df['class_id'] = df['class'].apply(lambda x: encoder[x])

    images = defaultdict(list)
    for row in df.itertuples():
        if row.detections == 'None':
            continue
        vehicles = [x.split(':') for x in row.detections.split('|')]
        vehicles = [(row.class_id, (int(x[0]), int(x[1]))) for x in vehicles]
        images[row.image] += vehicles

    rows = []
    for path, vehicles in images.items():
        for y in list(range(0, SIZE - size, step)) + [SIZE - size]:
            for x in list(range(0, SIZE - size, step)) + [SIZE - size]:
                # New image path
                directory = "data/training_sliding/%s/" % path.replace('.jpg', '')
                save_path = directory + path.replace('.', '_x%sy%s.' % (x, y))

                positions = []
                for class_id, (v_x, v_y) in vehicles:
                    if class_id == 8:
                        continue
                    if all([v_x >= x, v_x < x + size, v_y >= y, v_y < y + size]):
                        positions += [class_id, v_x - x, v_y - y,
                                      DIMS[class_id], DIMS[class_id]]

                # Save new image
                if save_images and positions:
                    try:
                        makedirs(directory)
                    except:
                        pass
                    img = Image.open("data/training/" + path)
                    img2 = img.crop((x, y, x + size, y + size))
                    img2.save(save_path)
                if positions:
                    rows += [[save_path] + positions]

    # Dataset to use for training
    df_out = pd.DataFrame(rows)
    df_out = df_out.rename_axis({0: 'path'}, axis="columns")
    return df_out


def main():
    df = pd.read_csv('data/trainingObservations.csv')
    df_out = extract_crops_sw(df, IMG_SIZE, True, 150)
    df_out.to_csv('data/training_cropped.csv')

if __name__ == '__main__':
    main()
