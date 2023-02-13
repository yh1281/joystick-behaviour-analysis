from matplotlib import pyplot
import scipy
##import statistics as stats
import numpy as np
import pandas as pd
import sysl

##Data extraction and reformat
data_path ='/Users/yilinhao/Downloads/js data_Yilin/661/task_autonomous_230127_661_1256.txt'
data=np.loadtxt(data_path,dtype="str")
data[0,0]=data[0,0].replace('\x00', '')
time=list(((data[:,0])).astype(int))
x=list(((data[:,1])).astype(float))
y=list(((data[:,2])).astype(float))
lock=list(((data[:,3])).astype(int))
tone=list(((data[:,4])).astype(int))
reward=list(((data[:,5])).astype(int))


'''data = pd.read_csv(data_path, sep="/t", header=None)
data.columns = ["time_ms", "x_mm","y_mm","jslock","toneon","rvalveopen","rewards_total",'o']
data.drop(columns=['o'],inplace=True)
data=data.dropna(axis=0)
data'''

##Find tone onset and offset
tone_onset = []
tone_offset = []
for n in range(0, len(tone)-1):
    if tone[n] == 0 and tone[n+1] == 1:
        tone_onset.append(n)
    elif tone[n] == 1 and tone[n+1] == 0:
        tone_offset.append(n)
tone_onset_time = []
tone_offset_time = []
for m in tone_onset:
    tone_onset_time.append(time[m])
tone_onset_time=np.array(tone_onset_time)

##Find joystick lock and unlock indices
js_lock_index = []
js_unlock_index = []
for n in range(0, len(lock)-1):
    if lock[n] == 0 and lock[n+1] == 1:
        js_lock_index.append(n)
    elif lock[n] == 1 and lock[n+1] == 0:
        js_unlock_index.append(n)
js_lock_index = js_lock_index[1:-1]
js_unlock_index = js_unlock_index[:-1]
##Find joystick lock and unlock times
js_lock_times = []
js_unlock_times = []
for j in js_lock_index:
    js_lock_times.append(time[j])
for s in js_unlock_index:
    js_unlock_times.append(time[s])

#@Fine reward given indices and times
reward_index = []
reward_time = []
for k in range(len(reward)-1):
    if reward[k] == 0 and reward[k+1] == 1:
        reward_index.append(k)
        reward_time.append(time[k])

##Find tone positive and tone negative joystick lock and unlock times
tone_pos_unlock_index = []
tone_neg_unlock_index = []
tone_pos_lock_index = []
tone_neg_lock_index = []
for g in tone_onset:
    for q in range(len(js_unlock_index)):
        if js_unlock_index[q]<=g<=js_lock_index[q]:
            tone_pos_unlock_index.append(js_unlock_index[q])
            tone_pos_lock_index.append(js_lock_index[q])
for d in js_unlock_index:
    if d not in tone_pos_unlock_index:
        tone_neg_unlock_index.append(d)
for f in js_lock_index:
    if f not in tone_pos_lock_index:
        tone_neg_lock_index.append(f)

##xy data preprocess
x = np.array(x)
x_baseline = np.mean(x[9978:10126])
x = x-x_baseline
y = np.array(y)
y_baseline = np.mean(y[9978:10126])
y = y-y_baseline
output_x = scipy.signal.savgol_filter(x, 51, 2)
output_y = scipy.signal.savgol_filter(y, 51, 2)

##Detect IGM
movement_onset_index_list = []
move_onset_latency_list= []
move_onset_time_list = []
for i in range(len(js_unlock_index)):
    start_point = js_unlock_index[i]
    end_point = js_lock_index[i]
    x_trajectory = np.array(output_x[start_point:end_point])
    y_trajectory = np.array(output_y[start_point:end_point])
    time_trajectory = np.array(time[start_point:end_point])
    hypotenuse = np.sqrt((x_trajectory**2)+(y_trajectory**2))
    for j in range(len(hypotenuse)):
            if hypotenuse[j]>=1.28:
                movement_radius=j
                break
            else:
                movement_radius=0
        hypotenuse_cut=np.array(hypotenuse[0:movement_radius])
        hypotenuse_cut=np.flip(hypotenuse_cut)
        time_cut=np.array(time_trajectory[0:movement_radius])
        time_cut=np.flip(time_cut)
        if movement_radius>0:
            for h in range(len(hypotenuse_cut)-1):
                if hypotenuse_cut[h]<=hypotenuse_cut[h+1]:
                    move_onset_time=time_cut[h]
                    move_onset_unlock = time_cut[-1]
                    break
            move_onset_latency = move_onset_time - move_onset_unlock
            move_onset_latency_list.append(move_onset_latency)
            move_onset_time_list.append(move_onset_time)

all_time_hypotenuse = np.sqrt((output_x ** 2) + (output_y ** 2))
pyplot.plot(time,all_time_hypotenuse)
pyplot.plot(time,lock)
pyplot.plot(time,tone)
#pyplot.plot(move_onset_time_list,np.zeros(len(move_onset_time_list)),'o')
pyplot.plot(time,reset_break,'o')
pyplot.hist(move_onset_latency_list, bins=np.linspace(0,2000,100))

for m in range(len(move_onset_latency_list)):
    if move_onset_latency_list[m]==0:
        print(m)


##Exclude lock and unlock events due to imbalances and premature movements
movement_related_unlock_index=[]
movement_related_lock_index=[]
for b in range(len(js_lock_index)):
    if time[js_lock_index[b]]-time[js_unlock_index[b]]>260:
        movement_related_lock_index.append(js_lock_index[b])
        movement_related_unlock_index.append(js_unlock_index[b])


movement_onset_time_list=[]
for c in movement_onset_index_list:
    movement_onset_time_list.append(time[c])
movement_onset_time_list=np.array(movement_onset_time_list)

movement_related_unlock_time=[]
for d in movement_related_unlock_index:
    movement_related_unlock_time.append(time[d])
movement_related_unlock_time=np.array(movement_related_unlock_time)
movement_onset_latency=movement_onset_time_list-movement_related_unlock_time

##Detect points when js crosses reset threshold
reset_break=(all_time_hypotenuse>1.28).astype(int)
reset_break_index = []
for w in range(len(all_time_hypotenuse)):
    if all_time_hypotenuse[w]>1.28:
        reset_break_index.append(w)
reset_break_index=np.array(reset_break_index)
##
engaged_tone = []
unengaged_tone = []
for i in range(len(tone_onset)):
    data=reset_break[tone_onset[i]:tone_offset[i]]
    if sum(data)>1:
        engaged_tone.append(tone_onset[i])
    else:
        unengaged_tone.append(tone_onset[i])

