import detect
from tkinter import filedialog




def read():
    #temp_path = input("输入文件路径")
    temp_path = select_image()
    notice_count = int(input("输入提示词数量"))

    if(notice_count <= 0):
        print("Invalid count")
        return
    array_input = []
    for i in range(0,notice_count):
        user_input = input(f"输入您的第{i+1}个参数")
        array_input.append(user_input)
    
    detect.class_demo1(temp_path,array_input)


def read_with_post(path, arraysignwords):
    return detect.class_demo_post(path,arraysignwords)

def strawberry_read_with_post(path):
    return detect.class_demo_strawberry_post(path)#提示词不从接口传入 改到后端。

def zeroshot_strawberry_test(path):
    return detect.zeroshot_strawberry_test(path)

def select_image():
    temp_path = filedialog.askopenfilename(
        title = "选择图片文件",
        filetypes=[("图片文件","*.jpg *.jpeg *.png *.bmp"),("所有文件","*.*")]
    )
    return temp_path



#if __name__ == '__main__':
    #detect.class_demo1()
    #CLIP_demo()
    #lass_demo2()
    #read()
    # strawberry_read_with_post("F:\\Model\\CLIPfile\\clip\\CLIP-main\\uploads\\123.jpg",
    #                                   ["Nutrient-deficient",
    #                                    "Strawberry Aphid",
    #                                    "Calcium-deficient",
    #                                     "plant with leaf spot disease",
    #                                     "strawberry plant with powdery mildew",
    #                                     "strawberry with disease but not Nutrient-deficient Strawberry Aphid Fruit-bearing or Calcium-deficient"
    #                                     "normal strawberry fruit",
    #                                     "normal strawberry plant",
    #                                     "not related to strawberry",
    #                                    ])