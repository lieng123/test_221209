from detectron2.engine import DefaultPredictor 
from detectron2.config import get_cfg 
from detectron2.data import MetadataCatalog 
from detectron2.utils.visualizer import ColorMode,Visualizer 
from detectron2 import model_zoo 
import xlsxwriter


import cv2 
import numpy as np 
import math 
from typing import List,Tuple,Union 
import torch 
from torch import device 
import re 
import os 
import xlrd #xlrd==1.2.0 
from PIL import Image 


stuff_classes=['things', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 
'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind', 'window', 'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor', 'pavement', 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'wall', 'rug']
thing_classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] 




stuff_dataset_id_to_contiguous_id={92: 1, 93: 2, 95: 3, 100: 4, 107: 5, 109: 6, 112: 7, 118: 8, 119: 9, 122: 10, 125: 11, 128: 12, 130: 13, 133: 14, 138: 15, 141: 16, 144: 17, 145: 18, 147: 19, 148: 20, 149: 21, 151: 22, 154: 23, 155: 24, 156: 25, 159: 26, 161: 27, 166: 28, 168: 29, 171: 30, 175: 31, 176: 32, 177: 33, 178: 34, 180: 35, 181: 36, 184: 
37, 185: 38, 186: 39, 187: 40, 188: 41, 189: 42, 190: 43, 191: 44, 192: 45, 193: 46, 194: 47, 195: 48, 196: 49, 197: 50, 198: 51, 199: 52, 200: 53, 0: 0}
thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

stuff_list=[]
thing_list=[] 

for type,value in stuff_dataset_id_to_contiguous_id.items():
    item=value
    stuff_list.append(item)
for type,value in thing_dataset_id_to_contiguous_id.items():
    item=value
    thing_list.append(item) 

#print(thing_list) 


path1 = "tester/" 
file_name_list = os.listdir(path1) 
#print(file_name_list)  
#print(len(file_name_list)) 


skc_answer = [] 
skc_rate = [] 
skc_mid = [] 
skc_max = [] 
skc_min = [] 
skc_size = [] 

#pos_dot/img_size
pdis = [] 
test_label_name =[] 
sim_label = [] 


y_num = 0 
y_list=[]
y_wid=[]
y_hei=[]
y_mid = [] 
y_class = []
y_class_num = [] 
y_thing_name = [] 
list_label = [] 
list_label_num = []
list_label_name = []
y_dot=[] 

data_i = 0

skc_answer_no_same = [] 
skc_answer_num = [] 



score_final_list = [] 

#user_input = input("Please input something: ")

#user_input_match_num = 0 
thing_num1 = 0 

""" for i in range(len(thing_classes)):
    if thing_classes[i] != user_input:
        thing_num1+=1 
    else:
        break
print(thing_num1," ",len(thing_classes))
if thing_num1+1>len(thing_classes):
    print("Is no input thing") """

path_excel = "excel/" 
file_excel_list = os.listdir(path_excel)
del(file_excel_list[0])
print(file_excel_list) 
file_excel_list2 = [] 


for i in range(len(file_excel_list)):
    exec("file_excel_list2.append('test%s.xlsx')"%(i+1))


print(file_excel_list2)








path2 = 'sketch model_final.xlsx'
def read_xlsx_skc(excel_path,sheet_num=0): #skc
    global skc_answer,skc_rate,skc_mid,skc_max,skc_min,skc_size
    excel_handle = xlrd.open_workbook(excel_path) 
    sheet = excel_handle.sheet_by_index(sheet_num) 
    
    skc_answer = sheet.col_values(3)[1:] 
    skc_rate = sheet.col_values(4)[1:] 
    skc_mid =  sheet.col_values(5)[1:] 
    skc_max = sheet.col_values(6)[1:] 
    skc_min = sheet.col_values(7)[1:] 
    skc_size = sheet.col_values(8)[1:] 

def read_xlsx_test(excel_path,sheet_num=0): #test
    global pdis 
    global test_label_name
    global sim_label 

    excel_handle = xlrd.open_workbook(excel_path)
    sheet = excel_handle.sheet_by_index(sheet_num) 
    pdis = sheet.col_values(7)[1:] #pos_dot/img_size
    test_label_name = sheet.col_values(0)[1:] #label_anme



path_skc = "skc/"
file_name_list_skc = os.listdir(path_skc)  

#print("lennnnn",len(file_name_list))



for i in range(len(file_name_list)):


    skc_score = 0
    read_xlsx_skc(path2)

    

    path_test3 = file_excel_list2[i] 
    print("\n\n")
    print(path_test3)
    read_xlsx_test("excel/"+path_test3)


    list_pt = [] #pids,test_label_name
    for j in range(len(pdis)):
        list_pt.append([test_label_name[j],pdis[j]])
    print(list_pt) 
    #[['person', 0.1688902798773831], ['car', 0.3946224516040984], ['airplane', 0.1836290168073402], ['suitcase', 0.2360641980557655], ['truck', 0.3467014216952213]]





    for j in range(len(skc_mid)): 
        skc_mid[j] = re.sub("[\(\)]", "", str(skc_mid[j]))
        skc_mid[j] = skc_mid[j].split(',') 
        skc_mid[j] = list(map(int,skc_mid[j]))
    print(skc_mid)
    #(472,407)
    #(236,203) 
    #skc_mid = list(map(int,skc_mid)) 



    """ skc_score = skc_dot*skc_size
    print("skc_socre",skc_score)  """
    #0.030400199891725314 

    for j in range(len(skc_answer)):
        for k in range(len(pdis)):
            if(skc_answer[j]==test_label_name[k]):
                sim_label.append(skc_answer[j])
    print(sim_label,"---sim_label") #두 xlsx 같은 label




    for j in range(len(skc_answer)): 
        if(skc_answer[j] not in skc_answer_no_same):
            skc_answer_no_same.append(skc_answer[j]) 


    print(skc_answer_no_same)
    for j in range(len(skc_answer_no_same)):
        skc_answer_num.append(0)
    for j in range(len(skc_answer_no_same)):
        for k in range(len(sim_label)):
            if(sim_label[k]==skc_answer_no_same[j]):
                skc_answer_num[j]+=1
    print("skc_answer_num",skc_answer_num) #두 xlsx 같은 label 의 각 물체 수량




    for j in range(len(file_name_list_skc)):
        skc_path = path_skc+file_name_list_skc[j] 
        skc_img = Image.open(skc_path) 
        skc_wid = skc_img.width 
        skc_hei = skc_img.height 
        print(skc_wid,",",skc_hei)

        skc_wid_hei = skc_wid*skc_hei 
        print("skc_mid: ",skc_mid)

        for k in range(len(skc_answer)): # 스케치 excel에서 얻은 물체의 수량 
            skc_score += (math.pow((skc_mid[k][0]-skc_wid/2),2)+math.pow((skc_mid[k][1]-skc_wid/2),2))*skc_size[k]/skc_wid_hei
        #                (스케치물체 x1-스케치중심x_mid)^2+(스케치물체 y1-스케치중심y_mid)^2) * 스케치 물체 size / (스케치width*스케치height)
        #                (스케치 물체 내적은 모두 더한다. )
        print("skc_score:",skc_score)

        list_pt2 = [] 
        for m in range(len(sim_label)):
            for n in range(len(list_pt)):
                if(list_pt[n][0]==sim_label[m]):
                    list_pt2.append(list_pt[n])
        #print(list_pt2)

        score_full = 0
        for k in range(len(list_pt2)):
            score_full += list_pt2[k][1]

        print("score_full_:",score_full) 
        score_final = math.fabs(skc_score-score_full) 
        skc_answer_num_all = 0
        for k in range(len(skc_answer_num)):
            skc_answer_num_all += skc_answer_num[k] 
        if skc_answer_num_all == 0:
            score_final+=1000000

        print("score_this_img: ",score_final) 
        score_final_list.append([file_name_list[i],score_final])
        score_final=0 
        skc_score = 0 

    skc_dot = ((math.pow((skc_mid[1][0]-skc_wid/2),2)+math.pow((skc_mid[1][1]-skc_hei/2),2))) / (skc_wid*skc_hei) 
    print("skc_dot: ",skc_dot)



    list_pt.clear()
    sim_label.clear()
    
    skc_answer_num.clear()
    list_pt2.clear()
    skc_score=0
    score_full=0
    

print("\n")
print(score_final_list)
score_final_list_sorted = score_final_list.sort(key=lambda x:x[1],reverse=False) 
print(score_final_list) 




workbook_score1 = xlsxwriter.Workbook('score/score1.xlsx') 
worksheet_score1 = workbook_score1.add_worksheet()  
worksheet_score1.set_column('A:CA',30) 
worksheet_score1.write(0,0,'image') 
worksheet_score1.write(0,1,'dot_score') 
for i in range(len(score_final_list)):
    worksheet_score1.write(i+1,0,score_final_list[i][0])
    worksheet_score1.write(i+1,1,score_final_list[i][1])


workbook_score1.close()

workbook_score2 = xlsxwriter.Workbook('score/score_5.xlsx') 
worksheet_score2 = workbook_score2.add_worksheet()  
worksheet_score2.set_column('A:CA',30) 
worksheet_score2.write(0,0,'image') 
worksheet_score2.write(0,1,'dot_score') 
for i in range(5):
    worksheet_score2.write(i+1,0,score_final_list[i][0])
    worksheet_score2.write(i+1,1,score_final_list[i][1])

workbook_score2.close()

class Detector:
    def __init__(self,model_type="OD"):
        self.cfg = get_cfg() 
        self.model_type = model_type 
        
        
        if(model_type == "OD"): 
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif(model_type == "IS"): 
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif(model_type == "Key"): 
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif(model_type == "PS"): 
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        elif(model_type == "PascalVOC"): 
            self.cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")
            
        elif(model_type == "LVIS"): 
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
        self.cfg.MODEL.DEVICE="cuda"  

        self.predictor=DefaultPredictor(self.cfg) 



    def onImage(self,imagePath):
        image=cv2.imread(imagePath)
        global y_num 
        global y_list
        global y_wid 
        global y_hei 
        global y_mid 
        global y_class
        global y_thing_name
        global data_i 
        global thing_num1 
        global user_input_match_num 
        global list_label 
        global list_label_num 
        global list_label_name
        global y_dot 
    

        if(self.model_type) == "OD" or "IS":
            predictions = self.predictor(image) 
            image_size1 = predictions['instances'].image_size[0]*predictions['instances'].image_size[1]
            #print(predictions['instances'].image_size[0]) 
            #print(predictions['instances'].image_size[1])
            


            #print(predictions['instances'].get_fields()['pred_classes'])
            list_classes = re.sub("[A-Za-z\()\[\]\ \n=':]", "", str(predictions['instances'].get_fields()['pred_classes']))
            
            list_classes = list_classes[:-2] 
            list_classes = list_classes.split(',') 
            list_int1 =list(map(int,list_classes)) 
            user_input_match_num = list_int1.count(thing_num1)
            print("user_input_match_num = ",user_input_match_num)

            print(list_int1) 
            #str에서 list에 원소외 다른 원소 제외하고 list 만들기 ↑ 
            list_label = [] 
            for i in list_int1:
                if(i not in list_label):
                    list_label.append(i) 
            for i in range(len(list_label)):
                list_label_name.append(thing_classes[i])
            #list_label => ['0', '1', '2'] 
            #print(list_label) 
            for i in range(len(list_label)):
                list_label_num.append(0)

            for i in range(len(list_label)):
                for j in range(len(list_int1)):
                    if(list_int1[j]==list_label[i]):
                        list_label_num[i]+=1

            print(list_label,"label")
            print(list_label_num,"label_num")

            



            """ if(list_label[-1]!='.'): 
                
                list_label = list(map(int,list_label))  """


            #list_label => [0, 1, 2]
            if(predictions['instances'].__len__()>0):

                workbook_label1 = xlsxwriter.Workbook('excel/label1.xlsx')
                worksheet_label1 = workbook_label1.add_worksheet() 
                worksheet_label1.set_column('A:CA',30)

                worksheet_label1.write(0,0,'image name') 
                worksheet_label1.write(1,0,'image width') 
                worksheet_label1.write(2,0,'image height') 
                worksheet_label1.write(3,0,'image mid') 
                worksheet_label1.write(4,0,'class') 
                worksheet_label1.write(5,0,'label_class') 
                worksheet_label1.write(6,0,'label_name')
                worksheet_label1.write(7,0,'label_num')

                workbook_sim1 = xlsxwriter.Workbook('sim1.xlsx') 
                worksheet_sim1 = workbook_sim1.add_worksheet() 





                
                
                
                """ worksheet_label1.write(1,i+1,predictions['instances'].image_size[0])
                worksheet_label1.write(2,i+1,predictions['instances'].image_size[1]) """
                


                for i in range(len(file_name_list)): #5 
                    
                    worksheet_label1.write(0,i+1,file_name_list[i]) 

                    
                    #print("asdassddsda",str(list_int1)) 
                    
                    
                


                    exec('workbook_data%s= xlsxwriter.Workbook("excel/test"+str(%s)+".xlsx")'%(i+1,i+1))
                    exec("worksheet_data%s= workbook_data%s.add_worksheet()"%(i+1,i+1))
                    exec("worksheet_data%s.set_column('B:B',30)"%(i+1))
                    exec("worksheet_data%s.set_column('F:F',30)"%(i+1))
                    #test.xlsx 5개 만들기 
                    exec("worksheet_data%s.write(0,0,'labelname')"%(i+1))
                    exec("worksheet_data%s.write(0,1,'position')"%(i+1))
                    exec("worksheet_data%s.write(0,2,'score')"%(i+1))
                    exec("worksheet_data%s.write(0,3,'size')"%(i+1)) 
                    exec("worksheet_data%s.write(0,4,'pos_mid')"%(i+1))
                    exec("worksheet_data%s.write(0,5,'pos_sub')"%(i+1)) 
                    exec("worksheet_data%s.write(0,6,'pos_dot')"%(i+1)) 
                    exec("worksheet_data%s.write(0,7,'pos_dot/img_size')"%(i+1)) 





                for j in range(len(list_classes)): #그림중 물체의 수량 
                    str_pos = re.sub("[A-Za-df-z\()=':]", "", str(predictions['instances'].get_fields()['pred_boxes'].__getitem__(j)))
                    #str_pos => ee[[100.7111, 482.9366, 235.9817, 628.8959]], ee0
                    str_pos=str_pos[2:-5]
                    #str_pos => [[100.7111, 482.9366, 235.9817, 628.8959]]
                    str_pos=re.sub("[\[\]]", "", str_pos) 
                    #str_pos => 100.7111, 482.9366, 235.9817, 628.8959
                    str_pos=str_pos.split(",")
                    #str_pos => ['100.7111', ' 482.9366', ' 235.9817', ' 628.8959']  (list)
                    str_pos = [float(k) for k in str_pos] 
                    #str2_pos => [100.7111, 482.9366, 235.9817, 628.8959] (list)
                    str_pos_int = list(map(int,str_pos)) 

                    str_xy1 = "["+str(str_pos_int[0])+","+str(str_pos_int[1])+"]"
                    str_xy2 = "["+str(str_pos_int[2])+","+str(str_pos_int[3])+"]"
                    str_final_xy = str_xy1+","+str_xy2
                    str_mid = "["+str(int(predictions['instances'].image_size[0]/2))+","+str(int(predictions['instances'].image_size[1]/2))+"]"

                    #print(str_final_xy) 
                    
                    str_id = list_classes[j] #list_classes 원소 지금 str이다.
                    #str_id => 0
                    str_label = thing_classes[int(str_id)] 
                    #str_label => person 
                    pos_area = (str_pos_int[2]-str_pos_int[0])*(str_pos_int[3]-str_pos_int[1])
                    pos_size = str(round(pos_area/image_size1*100,3))+"%" 
                    

                    str_score = re.sub("[A-Za-z\()=':]", "", str(predictions['instances'].get_fields()['scores'].__getitem__(j))) 
                    str_score = str_score[:-3] 

                    """ print("thing",str_id,": ",str_label,"\t","score: ",str_score,
                    "\t","position: ",str_pos)  """
                    #thing 0 :  person        score:  0.9972          position:  [100.7111, 482.9366, 235.9817, 628.8959]
                    pos_mid = "["+str(str_pos_int[2]-str_pos_int[0])+","+str(str_pos_int[3]-str_pos_int[1])+"]" 
                    pos_sub = "["+str((str_pos_int[2]-str_pos_int[0])-(predictions['instances'].image_size[0]/2))+","+str((str_pos_int[3]-str_pos_int[1])-(predictions['instances'].image_size[1]/2))+"]"
                    pos_dot = math.pow(((str_pos_int[2]-str_pos_int[0])-(predictions['instances'].image_size[0]/2)),2)+math.pow(((str_pos_int[3]-str_pos_int[1])-(predictions['instances'].image_size[1]/2)),2)
                    pos_weightxheight = (predictions['instances'].image_size[0]*predictions['instances'].image_size[1]) 
                    pos_image_thing_size = round(pos_area/image_size1*100,3)
                    
                    pos_dot_img = pos_dot/pos_weightxheight *pos_image_thing_size 
                    

                    if(float(str_score)>=0.9):
                        exec("worksheet_data%s.write(%s,0,str_label)"%(data_i+1,j+1)) 
                        exec("worksheet_data%s.write(%s,1,str_final_xy)"%(data_i+1,j+1)) 
                        exec("worksheet_data%s.write(%s,2,float(str_score))"%(data_i+1,j+1)) 
                        exec("worksheet_data%s.write(%s,3,pos_size)"%(data_i+1,j+1)) 
                        exec("worksheet_data%s.write(%s,4,pos_mid)"%(data_i+1,j+1))
                        exec("worksheet_data%s.write(%s,5,pos_sub)"%(data_i+1,j+1)) 
                        exec("worksheet_data%s.write(%s,6,pos_dot)"%(data_i+1,j+1)) 
                        exec("worksheet_data%s.write(%s,7,pos_dot_img)"%(data_i+1,j+1)) 

                 
                
                    
                y_list.append(str(list_int1)) 
                y_wid.append(predictions['instances'].image_size[0])
                y_hei.append(predictions['instances'].image_size[1]) 
                y_mid.append(str_mid)
                y_class.append(str(list_label))
                y_class_num.append(str(list_label_num))
                y_thing_name.append(str(list_label_name))


                print(y_num)
                y_num+=1 
                #y_list,y_wid,y_hei,  detectron가 한번 돌리면 데이터를 list에 append,
                #y_num(그림수량) n가 되면 list에 추가했던 n개 이미지의 데이터를 순서대로 excel에 write 
                 


                if(y_num==len(file_name_list)):
                    for i in range(len(file_name_list)): 
                        worksheet_label1.write(1,i+1,y_wid[i]) 
                        worksheet_label1.write(2,i+1,y_hei[i]) 
                        worksheet_label1.write(3,i+1,y_mid[i]) 
                        worksheet_label1.write(4,i+1,y_list[i]) 
                        worksheet_label1.write(5,i+1,y_class[i]) 
                        worksheet_label1.write(6,i+1,y_thing_name[i]) 
                        worksheet_label1.write(7,i+1,y_class_num[i]) 


                if(y_num==len(file_name_list)):
                    y_num=0
                
                
                exec("workbook_data%s.close()"%(data_i+1))  
                data_i+=1   
                
                
                workbook_label1.close()
            
            
            list_label_num.clear() 
            list_label_name.clear()

            #defalult,do not change ↓ 
            viz=Visualizer(image[:,:,::-1],metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode=ColorMode.IMAGE) 
            output=viz.draw_instance_predictions(predictions["instances"].to("cpu")) 

            image_sml = cv2.resize(output.get_image()[:,:,::-1],(1300,900)) 


            


            #image show
            """ cv2.imshow("Result",image_sml)
            cv2.waitKey(0) """


        elif(self.model_type) == "PS": 
            predictions,segmentInfo = self.pedictor(image)
            viz = Visualizer(image[:,:,::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])) 
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"),segmentInfo) 

            print(segmentInfo) 





detector = Detector(model_type = "OD") 

#print(path1+file_name_list[0]) 
#detector.onImage("tester/test1.jpg") 

for i in range(len(file_name_list)): 
    
    detector.onImage(path1+file_name_list[i]) 

 