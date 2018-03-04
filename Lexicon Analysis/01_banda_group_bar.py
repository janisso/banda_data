import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

sns.set(style="ticks")
sns.set()



colors =['#33ccff',
        '#3399ff',
        '#000099',
        '#660099',
        '#990099',
        '#cc0099',
        '#ff00cc',
        '#ff0033',
        '#ff6600',
        '#ff9900',
        '#ffbb00',
        '#ffdd00']


         
'''colors =['#c48f6c',#4
         '#c0622e',#10
         '#253c5b',#5
         '#dc423f',#1
         '#8f9190',#6
         '#5a2d36',#2
         '#5486c2',#9
         '#10585c']'''



sns.palplot(sns.color_palette(colors))
#colors = sns.color_palette("Paired", 10)
sns.set_palette(sns.color_palette(colors))



#from matplotlib import cm
#
#
#cs=cm.Set1(np.arange(40)/40.)
#f=plt.figure()
#ax=f.add_subplot(111, aspect='equal')
#p=plt.pie(a, colors=cs)
#plt.show()



#def do_a_pie(p1,axarr):



labels = np.array(["sample pack","random"	"patterns	" , "single characteres", "onomatopaic",	"phatic",
                   "metalinguistics", "someone", "political", "comic","statements","html/command	",
                   "emoticons","senseless",	"slogans","questions","bully","demanding", "egoic",
                   "complaint","love","humour","musical","support"])



#sample pack
#######"random"
#####"patterns"
#####"single characteres"
#####"onomatopaic"
#####"phatic"
#####"metalinguistics"
#####"someone"
#####"political"
#####"comic"
#####"statements"
#####"html/command"
#####"emoticons"
"senseless"
"slogans"
"questions"	
"bully"
"demanding"
"egoic"
"complaint"
"love"
"humour"
"musical"
"support"

def sums(array):
    adding = 0
    arry = array.astype(int)
    for i in range(len(array)):
        adding = adding + arry[i]
        print adding
    return adding

def big_pie(axarr,p1):
    
    random = np.sum(p1["random"])
    patterns = np.sum(p1["patterns"])
    char = np.sum(p1["single_characteres"])
    
    ono = np.sum(p1['onomatopaic'])
    #ono = sums(p1['onomatopaic'])
    #print ono
    phatic = np.sum(p1['phatic'])
    metal = np.sum(p1['metalinguistics'])
    someone = np.sum(p1['someone'])
    politic = np.sum(p1['political'])
    comic = np.sum(p1['comic'])
    statement = np.sum(p1['statements'])
    code = np.sum(p1['htmlcommand'])
    emot = np.sum(p1['emoticons'])
    
    '''senseless = np.sum(p1["senseless"])
    slogans = np.sum(p1["slogans"])
    questions = np.sum(p1["questions"])
    bully = np.sum(p1["bully"])
    demanding = np.sum(p1["demanding"])
    egoic = np.sum(p1["egoic"])
    complaint = np.sum(p1["complaint"])
    love = np.sum(p1["love"])
    humour = np.sum(p1["humour"])
    musical = np.sum(p1["musical"])
    support = np.sum(p1["support"])'''
    
    #for i in range(len(p1)):
    #    if sum(p1[i])<1:
    #        #print(i,sum(p1[i]))
    
    #labels = np.array(['Phatic','Metalinguistic','Someone','Political','Onomatopaic','Comic','Statements','Abstract','Code','Emoticons'])
    labels = np.array(['Random','Patterns','Single Character','Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons'])
    tot_len = len(p1)
    #print tot_len
    sizes = np.array([random, patterns, char, ono,phatic,metal,someone,politic,comic,statement,code,emot])
    #print sizes
    perc = sizes/tot_len
    #print perc
    #order = np.flip(np.argsort(perc),axis=0).astype(int)
    order = np.arange(12)
    print order
    #print perc[order]
    #print labels[order]
    #explode = np.full(10,0.1)
    #fig1, axarr = plt.subplots()
    patches = axarr.pie(perc[order],labels=labels[order],startangle=180,wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
    #axarr.legend(labels)
    axarr.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    my_circle=plt.Circle( (0,0), 0.4, color='white')
    #p=patches.gcf()
    axarr.add_artist(my_circle)    
    return perc,order



def small_pie(axarr,p1,order,t):
    '''ph = np.sum(p1['phatic'])
    ml = np.sum(p1['metalinguistical'])
    so = np.sum(p1['someone'])
    po = np.sum(p1['political'])
    on = np.sum(p1['onomatopaic'])
    co = np.sum(p1['comic'])
    st = np.sum(p1['statements'])
    ab = np.sum(p1['abstract'])
    ht = np.sum(p1['htmlcommand'])
    em = np.sum(p1['emoticons'])'''
    
    random = np.sum(p1["random"])
    patterns = np.sum(p1["patterns"])
    char = np.sum(p1["single_characteres"])
    
    
    ono = np.sum(p1['onomatopaic'])    
    phatic = np.sum(p1['phatic'])
    metal = np.sum(p1['metalinguistics'])
    someone = np.sum(p1['someone'])
    politic = np.sum(p1['political'])
    comic = np.sum(p1['comic'])
    statement = np.sum(p1['statements'])
    code = np.sum(p1['htmlcommand'])
    emot = np.sum(p1['emoticons'])
    
    for i in range(len(p1)):
        if sum(p1[i]).astype(int)<1:
            print(i,sum(p1[i]))
    
    #labels = 'Phatic','Metalinguistic','Someone','Political','Onomatopaic','Comic','Statements','Abstract','Code','Emoticons'
    labels = np.array(['Random','Patterns','Single Character','Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons'])
    tot_len = len(p1)
    #sizes = np.array([ph,ml,so,po,on,co,st,ab,ht,em])
    sizes = np.array([random, patterns, char, ono,phatic,metal,someone,politic,comic,statement,code,emot])
    perc = sizes/tot_len
    #explode = [0,0,0,0,0,0,0,0,0,0]#np.zeros(len(size))

    patches = axarr.pie(perc[order], startangle=180,wedgeprops = { 'linewidth' : 0.5, 'edgecolor' : 'white' })
    my_circle=plt.Circle( (0,0), 0.5, color='white')
    axarr.add_artist(my_circle)
    #axarr.legend(labels)
    axarr.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    axarr.set_xlabel(t)
    return perc



def theme_time(un,ind,p):
    #labels = 'Phatic','Metalinguistic','Someone','Political','Onomatopaic','Comic','Statements','Abstract','Code','Emoticons'
    labels = np.array(['Random','Patterns','Single Character','Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons'])
    f, axarr = plt.subplots(1,sharex=True)
    #axarr[0].set_yticks(np.arange(10), labels)
    axarr.set_yticks(np.arange(10))
    axarr.set_yticklabels(labels)
    arr = np.zeros(len(un)-1)
    for i in range(1,len(un)):
        m = np.argmax(np.array(list(un[i])))
        axarr.plot(ind[i],m,'.',color=colors[m])
        #if m != 7:
        #    arr[i-1]=m
        #    axarr[1].axvspan(i-0.5, i+0.5, facecolor=colors[m])
    axarr.set_title('Thematic Dispersion plot for Unique Phrases in Performance '+str(p))


    
def theme_time_o(un,ind,p,order,sample_pack):
    ranges = np.array([0])
    indices = np.where(np.diff(sample_pack) != 0)[0]+1
    ranges = np.hstack((ranges,indices))
    ranges = np.hstack((ranges,np.array([len(sample_pack)-1])))
    print ranges
    print ind[ranges]
    colorss = ['#cccccc', '#aaaaaa', '#888888', '#666666']
    #labels = ['Phatic','Metalinguistic','Someone','Political','Onomatopaic','Comic','Statements','Abstract','Code','Emoticons']
    labels = ['Random','Patterns','Single','Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons']
    labels = np.array(labels)[order]
    un = un[:,order]
    f, axarr = plt.subplots(1,sharex=True,figsize=(8,3))
    #axarr[0].set_yticks(np.arange(10), labels)
    axarr.set_yticks(np.arange(12))
    axarr.set_yticklabels(labels)
    ind[0]=0
    for i in range(1,len(ranges)):
        c = np.median(sample_pack[ranges[i-1]:ranges[i]]).astype(int)
        #print un[ranges[i-1]][0],un[ranges[i]][0]
        axarr.axvspan(ind[ranges[i-1]],ind[ranges[i]],color=colorss[c])
    arr = np.zeros(len(un)-1)
    for i in range(1,len(un)):
        m = np.argmax(un[i])
        axarr.plot(ind[i],m,'.',color=colors[m])
        #if m != 7:
        #    arr[i-1]=m
        #    axarr[1].axvspan(i-0.5, i+0.5, facecolor=colors[m])
    axarr.set_title('Thematic Dispersion plot for Unique Phrases in Performance '+str(p))
    axarr.set_xlabel('Message Number')
    sp1 = mpatches.Patch(color='#cccccc', label='SP1')
    sp2 = mpatches.Patch(color='#aaaaaa', label='SP2')
    sp3 = mpatches.Patch(color='#888888', label='SP3')
    sp4 = mpatches.Patch(color='#666666', label='SP4')
    plt.legend(handles=[sp1,sp2,sp3,sp4],bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    
    plt.rcParams['ytick.labelsize'] = 'small'
    axarr.set_xlim(ind[0],ind[-1])
    axarr.grid(False)
    #plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    return ranges



def convert_array(p):
    p_temp = np.zeros((len(p),12))
    for i in range(len(p)):
        for j in range(len(p[i])):
            #print i,j
            p_temp[i,j]=p[i][j]
    return p_temp


def get_perc(s,p1):
    s1p = np.sum(p1[s],axis=0)
    #return s1p/len(s)
    return s1p

def count_samples(sample_pack,p1):
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    for i in range(len(sample_pack)):
        if sample_pack[i] == 0:
            s1.append(i)
        if sample_pack[i] == 1:
            s2.append(i)
        if sample_pack[i] == 2:
            s3.append(i)
        if sample_pack[i] == 3:
            s4.append(i)
    s1perc = get_perc(s1,p1)
    s2perc = get_perc(s2,p1)
    s3perc = get_perc(s3,p1)
    s4perc = get_perc(s4,p1)
    
    return s1perc, s2perc, s3perc, s4perc


    


#-----------OBJECTIVE
path = '/Users/mb/Desktop/Janis.So/06_qmul/01_banda/new_data'
p_1 = np.genfromtxt(path+'/p1.csv',delimiter=',',names=True, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12))
p_2 = np.genfromtxt(path+'/p2.csv',delimiter=',',names=True, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12))
p_3 = np.genfromtxt(path+'/p3.csv',delimiter=',',names=True, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12))
p_4 = np.genfromtxt(path+'/p4.csv',delimiter=',',names=True, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12))
labels = ['Random','Patterns','Single Character','Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons']




#---------SUBJECTIVE
p_1sp = np.genfromtxt(path+'/p1.csv',delimiter=',',skip_header=1, usecols=(13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24))
p_2sp = np.genfromtxt(path+'/p2.csv',delimiter=',',skip_header=1, usecols=(13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24))
p_3sp = np.genfromtxt(path+'/p3.csv',delimiter=',',skip_header=1, usecols=(13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24))
p_4sp = np.genfromtxt(path+'/p4.csv',delimiter=',',skip_header=1, usecols=(13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24))

'''
    senseless = np.sum(p1["senseless"])
    slogans = np.sum(p1["slogans"])
    questions = np.sum(p1["questions"])
    bully = np.sum(p1["bully"])
    demanding = np.sum(p1["demanding"])
    egoic = np.sum(p1["egoic"])
    complaint = np.sum(p1["complaint"])
    love = np.sum(p1["love"])
    humour = np.sum(p1["humour"])
    musical = np.sum(p1["musical"])
    support = np.sum(p1["support"])
    
    
labels = ['Sensless','Slogans','Questions','Bullying','Demanding','Egoic','Complaint','Love','Humour','Musical','Support','Uncategorized']
'''

def print_stats(a):
    for i in range(len(a)):
        print int(a[i])


p1_sp = np.genfromtxt(path+'/p1.csv',delimiter=',',skip_header=1, usecols=(0)).astype(int)
p2_sp = np.genfromtxt(path+'/p2.csv',delimiter=',',skip_header=1, usecols=(0)).astype(int)
p3_sp = np.genfromtxt(path+'/p3.csv',delimiter=',',skip_header=1, usecols=(0)).astype(int)
p4_sp = np.genfromtxt(path+'/p4.csv',delimiter=',',skip_header=1, usecols=(0)).astype(int)

s11,s12,s13,s14 = count_samples(p1_sp,p_1sp)
s21,s22,s23,s24 = count_samples(p2_sp,p_2sp)
s31,s32,s33,s34 = count_samples(p3_sp,p_3sp)
s41,s42,s43,s44 = count_samples(p4_sp,p_4sp)

print_stats(s44)
#s2p = get_perc(s2,p_1)


all_tings = np.array(list(p_1)+list(p_2)+list(p_3)+list(p_4))



#all_tings = list(p_2)



plt.figure()
ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4,rowspan=4)
ax2 = plt.subplot2grid((5, 4), (4, 0))
ax3 = plt.subplot2grid((5, 4), (4, 1))
ax4 = plt.subplot2grid((5, 4), (4, 2))
ax5 = plt.subplot2grid((5, 4), (4, 3))



#f, axarr = plt.subplots(2, 2)
e_all,order = big_pie(ax1,all_tings)
e_1 = small_pie(ax2,p_1,order,'P1')
e_2 = small_pie(ax3,p_2,order,'P2')
e_3 = small_pie(ax4,p_3,order,'P3')
e_4 = small_pie(ax5,p_4,order,'P4')
ax1.set_title('Thematic Occurance for all Performances')
plt.tight_layout()



ind_1 = np.array(p1_un)[:,0].astype(int)
ind_2 = np.array(p2_un)[:,0].astype(int)
ind_3 = np.array(p3_un)[:,0].astype(int)
ind_4 = np.array(p4_un)[:,0].astype(int)



p_1_temp = convert_array(p_1)
p_2_temp = convert_array(p_2)
p_3_temp = convert_array(p_3)
p_4_temp = convert_array(p_4)



theme_time_o(p_1_temp,ind_1,1,order,p1_sp)
theme_time_o(p_2_temp,ind_2,2,order,p2_sp)
theme_time_o(p_3_temp,ind_3,3,order,p3_sp)
theme_time_o(p_4_temp,ind_4,4,order,p4_sp)



#theme_time(p_2,ind_2,2)
#theme_time(p_3,ind_3,3)
#theme_time(p_4,ind_4,4)



#f, axarr = plt.subplots(2, 2)
#e_1 = do_pie(axarr[0,0],p_1)
#e_2 = do_pie(axarr[0,1],p_2)
#e_3 = do_pie(axarr[1,0],p_3)
#e_4 = do_pie(axarr[1,1],p_4)
#
#f, axarr = plt.subplots()
#e_all = do_pie(axarr,np.array(all_tings))



labels = 'Random','Patterns','Single','Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons'
labels = np.array(labels)[order]
N = len(labels)
ind = np.arange(N)
width = 0.2



fig, ax = plt.subplots()



rects1 = ax.bar(ind,e_1[order],width)#,color='b')
rects2 = ax.bar(ind+width,e_2[order],width)#,color='g')
rects3 = ax.bar(ind+width*2,e_3[order],width)#,color='r')
rects4 = ax.bar(ind+width*3,e_4[order],width)#,color='m')



ax.set_ylabel('Percentage')
ax.set_xticks(ind+width*3/2)
ax.set_xticklabels(labels,rotation='90')
ax.legend((rects1[0],rects2[0],rects3[0],rects4[0]),('P1','P2','P3','P4'))
#ax.set_yticks(rotation='90')
ax.set_title('Comparison of Recurrent Thematic Occurence Between Performances')
plt.gcf().subplots_adjust(bottom=0.2)


#f, axarr = plt.subplots()



np.flip(np.argsort(np.array([4,2,3,6,1])),axis=0)



N = 4



rn = (e_1[order][0],e_2[order][0],e_3[order][0],e_4[order][0])
ptr= (e_1[order][1],e_2[order][1],e_3[order][1],e_4[order][1])
sch = (e_1[order][2],e_2[order][2],e_3[order][2],e_4[order][2])
on = (e_1[order][3],e_2[order][3],e_3[order][3],e_4[order][3])
ph = (e_1[order][4],e_2[order][4],e_3[order][4],e_4[order][4])
ml = (e_1[order][5],e_2[order][5],e_3[order][5],e_4[order][5])
som = (e_1[order][6],e_2[order][6],e_3[order][6],e_4[order][6])
pol = (e_1[order][7],e_2[order][7],e_3[order][7],e_4[order][7])
com = (e_1[order][8],e_2[order][8],e_3[order][8],e_4[order][8])
sta = (e_1[order][9],e_2[order][9],e_3[order][9],e_4[order][9])
code = (e_1[order][10],e_2[order][10],e_3[order][10],e_4[order][10])
emo = (e_1[order][11],e_2[order][11],e_3[order][11],e_4[order][11])



cumsums = np.zeros((12,4))
cumsums[0,:]=rn
cumsums[1,:]=cumsums[0,:]+ptr
cumsums[2,:]=cumsums[1,:]+sch
cumsums[3,:]=cumsums[2,:]+on
cumsums[4,:]=cumsums[3,:]+ph
cumsums[5,:]=cumsums[4,:]+ml
cumsums[6,:]=cumsums[5,:]+som
cumsums[7,:]=cumsums[6,:]+pol
cumsums[8,:]=cumsums[7,:]+com
cumsums[9,:]=cumsums[8,:]+sta
cumsums[10,:]=cumsums[9,:]+code
cumsums[11,:]=cumsums[10,:]+emo




cumsums[:,0]=cumsums[:,0]/cumsums[-1,0]
cumsums[:,1]=cumsums[:,1]/cumsums[-1,1]
cumsums[:,2]=cumsums[:,2]/cumsums[-1,2]
cumsums[:,3]=cumsums[:,3]/cumsums[-1,3]


ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

plt.figure(figsize=(6,4))
#plt.subplot(121,colspan=2)

#plt.figure()
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2,rowspan=2)

p1 = ax1.bar(ind, rn, width)
p2 = ax1.bar(ind, ptr, width, bottom=cumsums[0,:])
p3 = ax1.bar(ind, sch, width, bottom=cumsums[1,:])
p4 = ax1.bar(ind, on, width, bottom=cumsums[2,:])
p5 = ax1.bar(ind, ph, width, bottom=cumsums[3,:])
p6 = ax1.bar(ind, ml, width, bottom=cumsums[4,:])
p7 = ax1.bar(ind, som, width, bottom=cumsums[5,:])
p8 = ax1.bar(ind, pol, width, bottom=cumsums[6,:])
p9 = ax1.bar(ind, com, width, bottom=cumsums[7,:])
p10 = ax1.bar(ind, sta, width, bottom=cumsums[8,:])
p11 = ax1.bar(ind, code, width, bottom=cumsums[9,:])
p12 = ax1.bar(ind, emo, width, bottom=cumsums[10,:])

#plt.ylabel('Ratio of Themes in Performances')
ax1.set_title('Ratio of Recurrent Themes in Performances')
#ax1.set_xticks(ind, ('P1', 'P2', 'P3', 'P4'))
ax1.set_xticks(np.arange(4))
ax1.set_xticklabels(('P1', 'P2', 'P3', 'P4'))
ax1.set_ylim(0,1)
#plt.yticks(np.arange(0, 81, 10))
#plt.legend((p1[0], p2[0], p3[0], p4[0],p5[0],p6[0],p7[0],p8[0],p9[0],p10[0]), list(np.array(labels)[order]))
plt.legend((p12[0],p11[0],p10[0], p9[0], p8[0], p7[0],p6[0],p5[0],p4[0],p3[0],p2[0],p1[0]), list( np.flip( np.array(labels)[order],axis=0)),bbox_to_anchor=(1,1),loc=2)
#plt.tight_layout()
#plt.show()'''
