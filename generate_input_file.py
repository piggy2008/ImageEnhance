import os
from PIL import Image

gt_root = 'Dataset05/training_aug/groundtruth'
left_high_root = 'Dataset05/training/left_high'
right_low_root = 'Dataset05/training/right_low'

# file = open('Dataset05/train_file.txt', 'w')
def generate_input_list():
    file = open('Dataset05/train_file.txt', 'w')
    imgs = os.listdir(gt_root)

    for img in imgs:
        line = img[:-4]
        print (line)
        file.writelines(line + '\n')

    file.flush()

def split_big_image():
    imgs = os.listdir(gt_root)
    step = 300
    save_gt_path = 'Dataset05/training_aug/groundtruth'
    save_high_path = 'Dataset05/training_aug/left_high'
    save_low_path = 'Dataset05/training_aug/right_low'
    for img in imgs:
        name = img[:-4]
        image = Image.open(os.path.join(gt_root, img))
        image_high = Image.open(os.path.join(left_high_root, name + '.png'))
        image_low = Image.open(os.path.join(right_low_root, name + '.jpg'))

        w, h = image.size
        count = 0
        for i in range(0, w - step, step):
            for j in range(0, h - step, step):
                crop_image = image.crop((i, j, i + step, j + step))
                crop_image_high = image_high.crop((i, j, i + step, j + step))
                crop_image_low = image_low.crop((i, j, i + step, j + step))

                if not os.path.exists(save_gt_path):
                    os.makedirs(save_gt_path)

                if not os.path.exists(save_high_path):
                    os.makedirs(save_high_path)

                if not os.path.exists(save_low_path):
                    os.makedirs(save_low_path)

                save_path = os.path.join(save_gt_path, name + '_' + str(count) + img[-4:])
                save_path_high = os.path.join(save_high_path, name + '_' + str(count) + '.png')
                save_path_low = os.path.join(save_low_path, name + '_' + str(count) + '.jpg')
                crop_image.save(save_path)
                crop_image_high.save(save_path_high)
                crop_image_low.save(save_path_low)
                count += 1
        # file.writelinesine + '\n')

generate_input_list()
# split_big_image()

