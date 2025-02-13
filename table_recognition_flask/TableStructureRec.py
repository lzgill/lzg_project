import base64
import os
import cv2
from PIL.ImagePath import Path
from bs4 import BeautifulSoup
import numpy as np
from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.utils_table_recover import format_html, plot_rec_box_with_logic_info, plot_rec_box
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition

lineless_engine = LinelessTableRecognition()
wired_engine = WiredTableRecognition()
table_cls = TableCls()


# 表格识别
class TableProcessor:

    def lineless_process_image(self, img_path):
        image = cv2.imread(img_path)
        cls, elasp = table_cls(img_path)
        table_engine = lineless_engine
        #save_dir =Path()
        html, elasp, polygons, logic_points, ocr_res = table_engine(img_path)

        if polygons is None:
            print("polygons is None")
            return

        assert len(polygons) == len(logic_points), "polygons and logic_points must have the same length"

        for i, polygon in enumerate(polygons):
            polygon_array = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.rectangle(image, polygon_array[0], polygon_array[1], (0, 255, 0), 2)
            x, y = polygon_array[1]
            #logic_points
            start_row, end_row, start_col, end_col = logic_points[i]

            text = f"({start_row + 1}-{end_row + 1}), ({start_col + 1}-{end_col + 1})"
            # 获取文本大小和基线
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            #
            x, y = polygon_array[1]
            text_x = x - 30
            text_y = y - 30 + text_size[1]

            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        height, width = image.shape[:2]
        if height > 1400 and width > 1000:
            new_dim = (int(width * 0.4), int(height * 0.4))
            image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

        return image, html

    def wired_process_image(self, img_path):
        image = cv2.imread(img_path)
        cls, elasp = table_cls(img_path)
        # if cls == 'wired':
        table_engine = wired_engine

        html, elasp, polygons, logic_points, ocr_res = table_engine(img_path)
        if polygons is None:
            print("polygons is None")
            return

        assert len(polygons) == len(logic_points), "polygons and logic_points must have the same length"

        for i, polygon in enumerate(polygons):
            polygon_array = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.rectangle(image, polygon_array[0], polygon_array[1], (0, 255, 0), 2)
            x, y = polygon_array[1]
            # logic_points
            start_row, end_row, start_col, end_col = logic_points[i]

            text = f"({start_row + 1}-{end_row + 1}), ({start_col + 1}-{end_col + 1})"
            # 获取文本大小和基线
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            #
            x, y = polygon_array[1]
            text_x = x + 2
            text_y = y + 2 + text_size[1]

            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        height, width = image.shape[:2]
        if height > 1400 and width > 1000:
            new_dim = (int(width * 0.4), int(height * 0.4))
            image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
        return image, html

    def insert_border_style(self, table_html_str: str):
        style_res = """<meta charset="UTF-8"><style>td, th {border: 1px solid black;} 
        table {border-collapse: collapse;}</style>"""
        prefix_table, suffix_table = table_html_str.split("<body>")
        html_with_border = f"{prefix_table}<body>{style_res}{suffix_table}"
        return html_with_border

    def save_html(self, save_path, html):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)

    def write_html(self, save_html_path, table_html_str):
        if save_html_path:
            html_with_border = self.insert_border_style(table_html_str)
            self.save_html(save_html_path, html_with_border)




# image_path='./_upload_images/image_39.jpg'
# processor = TableProcessor()
#
# image = processor.lineless_process_image(image_path)

#
# for i, point in enumerate(logit_points):
#     start_row, end_row, start_col, end_col = point
#     print(f"框 {i+1} 的起始行是 {start_row+1}，终止行是 {end_row+1}，起始列是 {start_col+1}，终止列是 {end_col+1}")




##解析html
# soup = BeautifulSoup(html, 'html.parser')
# table = soup.find('table')
#
# rows = table.find_all('tr')
#
# cell_info = []
#
# for row_index, row in enumerate(rows):
#
#     cells = row.find_all(['td', 'th'])
#     col_index = 0
#     for cell in cells:
#
#         text = cell.get_text(strip=True)
#
#         if not text:
#             continue
#
#
#         rowspan = int(cell.get('rowspan', 1))
#         colspan = int(cell.get('colspan', 1))
#
#         cell_info.append({
#             'text': text,
#             'row': row_index + 1,
#             'col': col_index + 1
#         })
#
#         col_index += colspan
#
# for info in cell_info:
#     print(f"文本: '{info['text']}', 行: {info['row']}, 列: {info['col']}")