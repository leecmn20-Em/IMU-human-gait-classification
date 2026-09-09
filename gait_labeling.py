#%%
from tool_codes import configure
from tool_codes import GaitCSVManager, GaitCSVList, gaitCSVLabelGenerator, gaitCSVshow
from tool_codes import GaitDataManager

# 0=sit 1=standing 2=stand 3=sitting 4=walking 5=turning back

listsname="temp/lists"
name="temp/attempt 7 20250514/imu_trial_7"
name2=name+"_labeled"
obj=GaitCSVManager.load(name2)
obj.plot(show_rotation=False,period=[850,980],show_labels=False)


def data_preparation(sequence_length:int):
    global data, lists
    lists = GaitCSVList("temp/lists")
    data = GaitDataManager(num_classes=6, sequence_length=sequence_length)
    data.add(lists.normal)
    data.split()
    data.process()
    data.summary()
    data.train_data[0].plot_weight()
    
#data_preparation(sequence_length=100)
#lists=GaitCSVList(listsname)
#lists.summary()
# %%
