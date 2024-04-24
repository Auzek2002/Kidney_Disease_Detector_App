#Model Code:


# import cv2
# from PIL import ImageOps, Image
# import numpy as np
# import tensorflow as tf
# import splitfolders
# import os
# from keras.models import load_model
# from sklearn.metrics import confusion_matrix,classification_report
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# from tensorflow.keras import datasets,layers,Sequential
# from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,GlobalAveragePooling2D
# from keras.applications.vgg16 import VGG16
# from keras.models import Model
# import matplotlib.pyplot as plt
# import seaborn as sns

# # #Extracting the Data:

# # cyst = 'C:\\Users\\User\\Downloads\\archive\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\Cyst'
# # normal = 'C:\\Users\\User\\Downloads\\archive\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\Normal'
# # stone = 'C:\\Users\\User\\Downloads\\archive\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\Stone'
# # Tumor = 'C:\\Users\\User\\Downloads\\archive\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\Tumor'
# # print(f"Number of Normal Kidney Images in dataset: {len(normal)}")
# # print(f"Number of Cyst Kidney Images in dataset: {len(cyst)}")
# # print(f"Number of Stone Kidney Images in dataset: {len(stone)}")
# # print(f"Number of Tumor Kidney Images in dataset: {len(Tumor)}")

# # x = []
# # y = []
# # class_names = ['Cyst','Normal', 'Stone', 'Tumor']
# # img_size = 224
# # for i in class_names:
# #     folder_path = os.path.join('C:\\Users\\User\\Downloads\\archive\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone',i)
# #     for j in os.listdir(folder_path):
# #         img = cv2.imread(os.path.join(folder_path,j))
# #         img = cv2.resize(img,(img_size,img_size))
# #         x.append(img)
# #         y.append(i)


# # splitfolders.ratio("C:\\Users\\User\\Downloads\\archive\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone", output="dataset", seed=1337, ratio=(.8, .1, .1), group_prefix=None)

# #Data Augmentation:

# channels=3
# batch_size=25
# test_batch_size=32
# test_steps=1
# train_path = 'C:\\Users\\User\\Desktop\\Kidney_STC_Detector\\dataset\\train'
# test_path = 'C:\\Users\\User\\Desktop\\Kidney_STC_Detector\\dataset\\test'
# val_path = 'C:\\Users\\User\\Desktop\\Kidney_STC_Detector\\dataset\\val'

# train_gen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

# test_gen = ImageDataGenerator()

# val_datagen = ImageDataGenerator()

# train_generator = train_gen.flow_from_directory(directory=train_path, batch_size=batch_size,class_mode='categorical',target_size=(224,224), shuffle=True)

# test_generator = test_gen.flow_from_directory(directory=test_path, batch_size=test_batch_size,class_mode='categorical',target_size=(224,224), shuffle=False)

# val_generator = val_datagen.flow_from_directory(directory=val_path, batch_size=batch_size,class_mode='categorical',target_size=(224,224), shuffle=True)

# #Building The Neural Network:

# vgg = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
# for layer in vgg.layers:
#     layer.trainable = False

# def add_on_layers(class_num,model):
#   top_model=model.output
#   top_model = GlobalAveragePooling2D()(top_model)
#   top_model = Dense(1024,activation='relu')(top_model)
#   top_model = Dense(512,activation='relu')(top_model)
#   top_model = Dense(64,activation='relu')(top_model)
#   top_model = Dense(32,activation='relu')(top_model)
#   top_model = Dense(16,activation='relu')(top_model)
#   top_model = Dense(class_num,activation='softmax')(top_model)
#   return top_model

# head = add_on_layers(4,vgg)
# model = Model(inputs=vgg.input,outputs=head)

# #Compiling the Model:

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#               loss=tf.keras.losses.CategoricalCrossentropy(),
#               metrics= ['accuracy'])

# #Training The Model:

# history = model.fit(train_generator,validation_data=val_generator,epochs=10)

# #Model Evaluation:

# loss , acc = model.evaluate(test_generator)
# print(f'Accuracy on test data: {acc*100:.2f}%')

# #Confusion Matrix:

# def print_info( test_gen, preds, save_dir, subject ):
#     class_dict=test_gen.class_indices
#     labels= test_gen.labels
#     file_names= test_gen.filenames
#     error_list=[]
#     true_class=[]
#     pred_class=[]
#     prob_list=[]
#     new_dict={}
#     error_indices=[]
#     y_pred=[]
#     for key,value in class_dict.items():
#         new_dict[value]=key             # dictionary {integer of class number: string of class name}
#     # store new_dict as a text fine in the save_dir
#     classes=list(new_dict.values())     # list of string of class names
#     errors=0
#     for i, p in enumerate(preds):
#         pred_index=np.argmax(p)
#         true_index=labels[i]  # labels are integer values
#         if pred_index != true_index: # a misclassification has occurred
#             error_list.append(file_names[i])
#             true_class.append(new_dict[true_index])
#             pred_class.append(new_dict[pred_index])
#             prob_list.append(p[pred_index])
#             error_indices.append(true_index)
#             errors=errors + 1
#         y_pred.append(pred_index)

#     y_true= np.array(labels)
#     y_pred=np.array(y_pred)
#     if len(classes)<= 40:
#         # create a confusion matrix
#         cm = confusion_matrix(y_true, y_pred )
#         length=len(classes)
#         if length<8:
#             fig_width=8
#             fig_height=8
#         else:
#             fig_width= int(length * .5)
#             fig_height= int(length * .5)
#         plt.figure(figsize=(fig_width, fig_height))
#         sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
#         plt.xticks(np.arange(length)+.5, classes, rotation= 90)
#         plt.yticks(np.arange(length)+.5, classes, rotation=0)
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.title("Confusion Matrix")
#         plt.show()
#     clr = classification_report(y_true, y_pred, target_names=classes)
#     print("Classification Report:\n----------------------\n", clr)

# sns.set_style('dark')
# p = model.predict(test_generator)
# print_info( test_generator, p, r'./', 'kidney')

# #Saving the Model:

# model.save('Kidney_stc_tf.h5')