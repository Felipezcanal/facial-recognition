# import os
# from shutil import copyfile
#
#
# def trav_dir(dirpath):
#     os.chdir(dirpath)
#     emotions_list = os.listdir()
#
#     for emotion in emotions_list:
#         os.chdir(emotion)
#         emojis_list = os.listdir()
#
#         for emoji in emojis_list:
#             os.chdir(emoji)
#             ids_list = os.listdir()
#
#             for id in ids_list:
#                 os.chdir(id)
#                 f = open("label.txt", "w+")
#                 f.write("%d" % int(emotion))
#                 f.close()
#                 os.chdir('./..')
#
#             os.chdir('./..')
#
#         os.chdir('./..')
#
#     os.chdir('./..')
#
# trav_dir("dataset/nosso_alinhado")



#################################################################################################################
# MUGGGG
#################################################################################################################

import os
from shutil import copyfile, rmtree

emotionsLabels = {'neutral':0, 'anger':1, 'contempt':2, 'disgust':3, 'fear':4, 'happiness':5, 'sadness':6, 'surprise':7}
def trav_dir(dirpath):
    os.chdir(dirpath)
    individuos = os.listdir()

    for individuo in individuos:
        os.chdir(individuo)
        emotions = os.listdir()

        for emotion in emotions:
            if (emotion == 'mixed'):
                rmtree(emotion)
                continue
            os.chdir(emotion)
            takes = os.listdir()

            for take in takes:
                print(individuo, emotion, take)
                os.chdir(take)
                f = open("label.txt", "w+")
                f.write("%d" % int(emotionsLabels[emotion]))
                f.close()
                os.chdir('./..')

            os.chdir('./..')

        os.chdir('./..')

    os.chdir('./..')

trav_dir("/app/mug")
