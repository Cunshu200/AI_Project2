import os, sys
# dirPath = "test/"
# print '移除前test目录下有文件：%s' %os.listdir(dirPath)
# #判断文件是否存在
# if(os.path.exists(dirPath+"foo.txt")):
# 　　os.remove(dirPath+"foo.txt")
# 　　print '移除后test 目录下有文件：%s' %os.listdir(dirPath)
# else:
# 　　print "要删除的文件不存在！"

DIR_NAME = './data/rollout/'
DIR_NAME = './data/series/'
DIR_NAME = "./data/GAN_series/"

# 要保留多少个文件这里就写几，全部删除就写0
# REMAIN_FILE_COUNT = 0

filelist = os.listdir(DIR_NAME)
print(len(filelist))
# for i, filename in enumerate(filelist):
#     if i >= REMAIN_FILE_COUNT:
#         os.remove(DIR_NAME + filename)

filelist = os.listdir(DIR_NAME)
print(len(filelist))