import sys
sys.path.insert(0, './YOLOX')
import torch
import numpy as np
import cv2
import time
from loguru import logger

from yolox.data.data_augment import preproc, ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp.build import get_exp_by_name, get_exp_by_file, get_exp
from yolox.utils import postprocess
from yolox.utils.visualize import vis



COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)




class Detector():
    """ 图片检测器 """
    def __init__(self, model='yolox-s', ckpt='weights/best_ckpt.pth'):
        super(Detector, self).__init__()



        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.exp = get_exp('./exps/example/custom/yolox_s.py')
        self.test_size = self.exp.test_size  # TODO: 改成图片自适应大小
        self.model = self.exp.get_model()
        self.num_classes = self.exp.num_classes
        self.nmsthre = self.exp.nmsthre
        self.model.to(self.device)
        self.model.cuda()
        self.model.half()  # to FP16
        self.model.eval()
        checkpoint = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.preproc = ValTransform(legacy=False)



    # def detect(self, raw_img, visual=True, conf=0.5):
    #     img_info = {"id": 0}
    #     img_info["file_name"] = None

    #     height, width = raw_img.shape[:2]
    #     img_info["height"] = height
    #     img_info["width"] = width
    #     img_info["raw_img"] = raw_img

    #     ratio = min(self.test_size[0] / raw_img.shape[0], self.test_size[1] / raw_img.shape[1])
    #     img_info["ratio"] = ratio

    #     raw_img, _ = self.preproc(raw_img, None, self.test_size)
    #     img = torch.from_numpy(raw_img).unsqueeze(0)
    #     img = img.float()
    #     img = img.cuda()
    #     img = img.half()  # to FP16

    #     with torch.no_grad():
    #         t0 = time.time()
    #         outputs = self.model(img)
    #         # if self.decoder is not None:
    #         #     outputs = self.decoder(outputs, dtype=outputs.type())
    #         outputs = postprocess(
    #             outputs, self.num_classes, conf,
    #             self.nmsthre, class_agnostic=True
    #         )
    #         logger.info("Infer time: {:.4f}s".format(time.time() - t0))
    #     print("outputs", outputs)
    #     print("img_info", img_info)

    #     # preprocessing: resize
        
    #     img_info['ratio'] = ratio
    #     return outputs, img_info
        # info = {}
        # # img, ratio = preproc(raw_img, self.test_size, COCO_MEAN, COCO_STD)
        # print("test size", self.test_size)
        # img, ratio = self.preproc(raw_img, None, self.test_size)
        # info['raw_img'] = raw_img
        # info['img'] = img

        # img = torch.from_numpy(img).unsqueeze(0)
        # img = img.to(self.device)

        # with torch.no_grad():
        #     outputs = self.model(img)
        #     outputs = postprocess(
        #         outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre, class_agnostic=True  # TODO:用户可更改
        #     )
            
        # if outputs[0] is None:
        #     info['boxes'], info['scores'], info['class_ids'],info['box_nums']=None,None,None,0
        # else:
        #     print(outputs[:][0:4]/ratio)
        #     info['boxes'] = outputs[:, 0:4]/ratio
        #     info['scores'] = outputs[:, 4] * outputs[:, 5]
        #     info['class_ids'] = outputs[:, 6]
        #     info['box_nums'] = outputs.shape[0]
        # # 可视化绘图
        # if visual:
        #     info['visual'] = vis(info['raw_img'], info['boxes'], info['scores'], info['class_ids'], conf, COCO_CLASSES)
        # return info
    def detect(self, raw_img, visual=True, conf=0.5):
        info = {}
        img, ratio = preproc(raw_img, self.test_size)
        info['raw_img'] = raw_img
        info['img'] = img

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.cuda()
        img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre, class_agnostic=True  # TODO:用户可更改
            )
        output = outputs[0]
        if output is None:
            info['boxes'], info['scores'], info['class_ids'],info['box_nums']=None,None,None,0
        else:
            info['boxes'] = output[:, 0:4]/ratio
            info['scores'] = output[:, 4] * output[:, 5]
            info['class_ids'] = output[:, 6]
            info['box_nums'] = output.shape[0]
            info['ratio'] = ratio
        # 可视化绘图
        # if visual:
        #     info['visual'] = vis(info['raw_img'], info['boxes'], info['scores'], info['class_ids'], conf, COCO_CLASSES)
        return info






if __name__=='__main__':
    detector = Detector()
    img = cv2.imread('dog.jpg')
    img_,out = detector.detect(img)
    print(out)
