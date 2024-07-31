# my_train

import train

params = {'weights': 'yolov5s6.pt',
          'cfg': 'models/hub/yolov5s6.yaml',
          'data': 'data/UNIMIB2016.yaml',
          'hyp': 'data/hyps/hyp.scratch-low.yaml',
          'epochs': 300,
          'batch_size': 2,
          'imgsz': 320,
          'optimizer': 'Adam',
          'cache': 'disk' }

if __name__ == '__main__':
    train.run(**params)



