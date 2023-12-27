
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import json
class blip_model():
    def __init__(self) -> None:
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    def gen_res(self,img_path):
        raw_image = Image.open(img_path).convert('RGB')

        text = "a people in front of "
        input_1 = self.processor(raw_image, text, return_tensors="pt").to("cuda")
        out_1 = self.model.generate(**input_1,max_length=100)
        res_1=self.processor.decode(out_1[0], skip_special_tokens=True)

        input_2 = self.processor(raw_image, return_tensors="pt").to("cuda")
        out_2 = self.model.generate(**input_2,max_length=100)
        res_2=self.processor.decode(out_2[0], skip_special_tokens=True)
        return res_1+". "+res_2
def gen_json(img_path,n): #使用Blip模型标注图片
    model=blip_model()
    #img_path="D:/NNDL/data/deepfashion-multimodal/images"
    #获取该目录下所有文件，存入列表中
    imgs=os.listdir(img_path)
    res={}
    start=31000

    for img in range(start,len(imgs)):
        img_k=imgs[img]
        img_path_=img_path+"/"+img_k
        res[img_k]=model.gen_res(img_path_)
        if len(res)>=n:
            break
    #保存为json文件
    with open('res.json', 'w') as f:
        json.dump(res, f,indent=2)
#print(model.gen_res("test.JPG"))
img_path="D:/NNDL/data/img"
#gen_json(img_path,5000)
print("@@@@")
def read_json(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    return data
#data=read_json("res.json")
#重新保存
def save_json(data,json_path):
    with open(json_path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2)
#save_json(data,"res_new.json")