'''
用于仅返回排序好的识别结果
'''
import os

from Fsk.view2 import app
from argparse import ArgumentParser
#from waitress import serve

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', default=2000)
    args = parser.parse_args()
    # 端口号可以设置本地和服务端，可以允许别人调用
    app.run(host='0.0.0.0', port=int(args.port), debug=True)
