from matplotlib import pyplot
import scipy
##import statistics as stats
import numpy as np
import pandas as pd
import sysl
import math
import chardet
##Data extraction and reformat
data_path ='C:/Users/Yilin Hao/Downloads/large_rwzone_230310_8212_1213.txt'

data=np.loadtxt(data_path,dtype="str")

data[868,0]=data[868,0].replace('\x00', '')
time=list(((data[:,0])).astype(int))
x=list(((data[:,1])).astype(float))
y=list(((data[:,2])).astype(float))
lock=list(((data[:,3])).astype(int))
tone=list(((data[:,4])).astype(int))
reward=list(((data[:,5])).astype(int))
rewardsum=list(((data[:,6])).astype(int))
segment2 = time[279235:362353]
for i, t in enumerate(segment2):
    segment2[i] = t-1000+502864


for i, t in enumerate(rewardsum[279233:362353]):
    rewardsum[279233:362353][i] = t+rewardsum[279232]



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
js_lock_index = js_lock_index[:]
js_unlock_index = js_unlock_index[:]
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
x_baseline = np.mean(x[0:200])
x = x-x_baseline
y = np.array(y)
y_baseline = np.mean(y[0:200])
y = y-y_baseline
output_x = scipy.signal.savgol_filter(x, 51, 2)
output_y = scipy.signal.savgol_filter(y, 51, 2)


##Detect movement onset
movement_onset_index_list = []
move_onset_latency_list= []
move_onset_time_list = []
for i in range(len(tone_neg_unlock_index)):
    start_point = tone_neg_unlock_index[i]
    end_point = tone_neg_lock_index[i]
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
pyplot.plot(time,all_time_hypotenuse,'o')
pyplot.plot(time,lock)
pyplot.plot(time,tone)
pyplot.plot(time,reward)
#pyplot.plot(move_onset_time_list,np.zeros(len(move_onset_time_list)),'o')
pyplot.plot(time,reset_break,'o')
pyplot.hist(move_onset_latency_list, bins=np.linspace(0,3000,100))


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
##Detect engaged and non-engaged tone
engaged_tone = []
unengaged_tone = []
for i in range(len(tone_onset)):
    data=reset_break[tone_onset[i]:tone_offset[i]]
    if sum(data)>1:
        engaged_tone.append(tone_onset[i])
    else:
        unengaged_tone.append(tone_onset[i])

## Detect rewarded tone and non-rewarded ton
rewarded_tone = []
rewarded_tone_offset = []
nonrewarded_tone = []
nonrewarded_tone_offset = []
for s in range(len(tone_onset)):
    data=reward[tone_onset[s]:tone_pos_lock_index[s]]
    if sum(data)>=1:
        rewarded_tone.append(tone_onset[s])
        rewarded_tone_offset.append(tone_offset[s])
    else:
        nonrewarded_tone.append(tone_onset[s])
        nonrewarded_tone_offset.append(tone_offset[s])


## hypotenuse = np.sqrt(x_traj_post_tone**2 + y_traj_post_tone**2)
##angle = np.arctan(y_traj_post_tone/x_traj_post_tone)
        pyplot.plot(x_traj_post_tone, y_traj_post_tone)
        pyplot.plot(x_traj_post_tone, y_traj_post_tone, 'o', markersize=1.5)
        pyplot.xlim(-7, 7)

fig = pyplot.figure(figsize=(9, 9))
pyplot.subplots_adjust(wspace=0.5,hspace=0.5)
for idx,t in enumerate(tone_onset[101:201]):
    ax = pyplot.subplot(10, 10, idx + 1)
    start_time = tone_onset[idx]
    if start_time in rewarded_tone:
        end_time = start_time+35
        x_traj_post_tone = np.array(x[start_time:end_time])
        y_traj_post_tone = np.array(y[start_time:end_time])
       ## hypotenuse = np.sqrt(x_traj_post_tone**2 + y_traj_post_tone**2)
        ##angle = np.arctan(y_traj_post_tone/x_traj_post_tone)
        pyplot.plot(x_traj_post_tone,y_traj_post_tone)
        pyplot.plot(x_traj_post_tone, y_traj_post_tone,'o',markersize=1.5)
        pyplot.xlim(-8, 8)
        pyplot.ylim(-1, 8)
        #pyplot.axhline(3,color="g")
        #pyplot.axhline(7,color="g")
        circle1=pyplot.Circle((0, 0), 3, color='g', fill=False)
        circle2=pyplot.Circle((0,0),7,color='g', fill=False)
        ax.add_patch(circle1)
        ax.add_patch(circle2)

    else:
        end_time = tone_offset[idx]
        x_traj_post_tone = np.array(x[start_time:end_time])
        y_traj_post_tone = np.array(y[start_time:end_time])
        pyplot.plot(x_traj_post_tone, y_traj_post_tone)
        pyplot.plot(x_traj_post_tone, y_traj_post_tone, 'o', markersize=1.5)
        pyplot.xlim(-8, 8)
        pyplot.ylim(-1, 8)
        #pyplot.axhline(3,color="r")
        #pyplot.axhline(7,color="r")
        circle3=pyplot.Circle((0, 0), 3,color='r', fill=False)
        circle4=pyplot.Circle((0, 0), 7,color='r', fill=False)
        ax.add_patch(circle3)
        ax.add_patch(circle4)

fig2=pyplot.figure()
pyplot.plot(all_time_hypotenuse)
pyplot.plot(all_time_hypotenuse,'o')
pyplot.plot(tone)
pyplot.plot(reward)
pyplot.plot(y,'o')