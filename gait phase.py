#%%
from tool_codes import configure
from tool_codes import Gait_data, Gait_model, GaitDataManager

#--------------------------------------------------
data=GaitDataManager(num_classes=6,sequence_length=100)
data.add('temp/attempt 2 20250429/imu_trial_1_labeled.csv')
data.train_names=data.names.copy()
data.val_names=data.names.copy()
data.test_names=data.names.copy()
data.process()
data.summary()
data.data[0].summary()

model=Gait_model(example_data=data.data[0],num_layer=100,dropout=0.25)
model.model.summary()

model.fit(train=data.train_data[0],val=data.val_data[0],epochs=5,batch_size=128,shuffle=True)
model.save('saved_models','gait_model')

model.test(data=data.test_data[0],show=True,plot=True)
# %%
