import os

from PIL import Image, ImageDraw



def read_boxes_from_file(file_path):
    """
    从文件中读取文本框的坐标。
    参数:
    - file_path: 文本文件的路径。
    返回:
    - boxes: 所有文本框的坐标列表。
    """
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            # 假设每行坐标由逗号分隔
            coords = line.strip().split(',')
            if len(coords) == 8:
                # 将字符串坐标转换为整数，并作为元组添加到列表中
                box = tuple(map(int, coords))
                boxes.append(box)
    return boxes


def calculate_bounding_box(rect):
    """
    计算由四个点定义的文本框的最小外接矩形的坐标。
    参数:
    - rect: 文本框的四个点的坐标，格式为 (x1, y1, x2, y2, x3, y3, x4, y4)。
    返回:
    - 外接矩形的坐标，格式为 (left, top, right, bottom)。
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = rect
    left = min(x1, x2, x3, x4)
    top = min(y1, y2, y3, y4)
    right = max(x1, x2, x3, x4)
    bottom = max(y1, y2, y3, y4)
    return left, top, right, bottom


def draw_boxes_on_image(image_path, boxes, output_path):
    """
    在图片上绘制多个文本框的坐标矩形，并保存结果图片。

    参数:
    - image_path: 要绘制的图片的路径。
    - boxes: 所有文本框的坐标列表，每个坐标格式为 (x1, y1, x2, y2, x3, y3, x4, y4)。
    - output_path: 绘制后的图片保存的路径。
    """
    # 打开图片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 遍历每个文本框
    for box in boxes:
        # 计算外接矩形
        left, top, right, bottom = calculate_bounding_box(box)

        # 绘制矩形框
        draw.rectangle([(left, top), (right, bottom)], outline='green', width=1)

    # 保存绘制后的图片
    img.save(output_path)


# def crop_image_from_boxes(image_path, boxes, output_folder):
#     """
#     根据多个文本框的坐标裁剪图片，并保存裁剪后的图片。
#
#     参数:
#     - image_path: 要裁剪的图片的路径。
#     - boxes: 所有文本框的坐标列表，每个坐标格式为 (x1, y1, x2, y2, x3, y3, x4, y4)。
#     - output_folder: 裁剪后的图片保存的文件夹路径。
#     """
#     # 打开图片
#     img = Image.open(image_path)
#
#     # 遍历每个文本框
#     for i, box in enumerate(boxes):
#         # 计算外接矩形
#         left, top, right, bottom = calculate_bounding_box(box)
#
#         # 裁剪图片
#         cropped_img = img.crop((left, top, right, bottom))
#
#         # 保存裁剪后的图片
#         output_path = f"{output_folder}/crop_{i + 1}.jpg"
#         cropped_img.save(output_path)
#         print(f"Saved cropped image to {output_path}")


def process_images(image_folder_path, txt_folder_path, output_folder, num_images=20):
    """
    处理文件夹中的指定数量的图片，并在每张图片上绘制文本框坐标的矩形框。

    参数:
    - image_folder_path: 包含图片的文件夹路径。
    - txt_folder_path: 包含文本框坐标的文本文件路径。
    - output_folder: 处理后的图片保存的文件夹路径。
    - num_images: 要处理的图片数量，默认为20。
    """
    # 获取文件夹中所有图片的路径
    image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]
    image_files.sort()  # 可选：按文件名排序

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 处理前num_images张图片
    for i, image_file in enumerate(image_files[:num_images]):
        # 构造对应的txt文件名
        txt_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        txt_file_path = os.path.join(txt_folder_path, txt_file)

        # 读取文本框坐标
        boxes = read_boxes_from_file(txt_file_path)

        # 构造图片路径和输出路径
        image_path = os.path.join(image_folder_path, image_file)
        output_path = os.path.join(output_folder, f"box_{i + 1}.jpg")

        # 在图片上绘制文本框
        draw_boxes_on_image(image_path, boxes, output_path)



# 示例使用
image_path = 'D:/dbnet-crnn/test_images/db_train_data/image'  # 替换为你的图片路径
txt_file_path = 'D:/dbnet-crnn/test_images/db_train_data/label'
output_folder = 'D:/dbnet-crnn/test_images/test_output'
output_path = 'D:/dbnet-crnn/test_images/test_output'# 替换为输出文件夹的路径
#boxes = read_boxes_from_file(txt_file_path)
#crop_image_from_boxes(image_path, boxes, output_folder)
process_images(image_path, txt_file_path, output_folder)



