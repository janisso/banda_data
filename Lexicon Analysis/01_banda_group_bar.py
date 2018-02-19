import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set()


colors =['#dc423f',#1
         '#5a2d36',#2
         '#eacdd5',#3
         '#c48f6c',#4
         '#253c5b',#5
         '#8f9190',#6
         '#10585c',#7
         '#9b9646',#8
         '#5486c2',#9
         '#c0622e']#10
         
colors =['#c48f6c',#4
         '#c0622e',#10
         '#253c5b',#5
         '#dc423f',#1
         '#8f9190',#6
         '#5a2d36',#2
         '#5486c2',#9
         '#10585c']
sns.palplot(sns.color_palette(colors))
colors = sns.color_palette("Paired", 10)
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

def big_pie(axarr,p1):
    
    '''random = np.sum(p1["random"])
    patterns = np.sum(p1["patterns"])
    char = np.sum(p1["single characteres"])'''
    
    
    ono = np.sum(p1['onomatopaic'])    
    phatic = np.sum(p1['phatic'])
    metal = np.sum(p1['metalinguistical'])
    someone = np.sum(p1['someone'])
    politic = np.sum(p1['political'])
    comic = np.sum(p1['comic'])
    statement = np.sum(p1['statements'])
    code = np.sum(p1['"html/command"'])
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
    
    

    
    for i in range(len(p1)):
        if sum(p1[i])<1:
            print(i,sum(p1[i]))
    
    #labels = np.array(['Phatic','Metalinguistic','Someone','Political','Onomatopaic','Comic','Statements','Abstract','Code','Emoticons'])
    labels = np.array(['Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons'])
    tot_len = len(p1)
    sizes = np.array([ono,phatic,metal,someone,politic,comic,statement,code,emot])
    perc = sizes/tot_len
    order = np.flip(np.argsort(perc),axis=0).astype(int)
    #print perc[order]
    #print labels[order]
    #explode = np.full(10,0.1)
    #fig1, axarr = plt.subplots()
    patches = axarr.pie(perc[order],labels=labels[order],startangle=180,wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' })
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
    
    
    ono = np.sum(p1['onomatopaic'])    
    phatic = np.sum(p1['phatic'])
    metal = np.sum(p1['metalinguistical'])
    someone = np.sum(p1['someone'])
    politic = np.sum(p1['political'])
    comic = np.sum(p1['comic'])
    statement = np.sum(p1['statements'])
    code = np.sum(p1['"html/command"'])
    emot = np.sum(p1['emoticons'])
    
    for i in range(len(p1)):
        if sum(p1[i])<1:
            print(i,sum(p1[i]))
    
    #labels = 'Phatic','Metalinguistic','Someone','Political','Onomatopaic','Comic','Statements','Abstract','Code','Emoticons'
    labels = np.array(['Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons'])
    tot_len = len(p1)
    #sizes = np.array([ph,ml,so,po,on,co,st,ab,ht,em])
    sizes = np.array([ono,phatic,metal,someone,politic,comic,statement,code,emot])
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
    labels = np.array(['Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons'])
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
    
def theme_time_o(un,ind,p,order):
    #labels = ['Phatic','Metalinguistic','Someone','Political','Onomatopaic','Comic','Statements','Abstract','Code','Emoticons']
    labels = ['Onomatopaic','Phatic','Metalinguistic','Someone','Political','Comic','Statements','Code','Emoticons']
    labels = np.array(labels)[order]
    un = un[:,order]
    f, axarr = plt.subplots(1,sharex=True,figsize=(8,4))
    #axarr[0].set_yticks(np.arange(10), labels)
    axarr.set_yticks(np.arange(10))
    axarr.set_yticklabels(labels)
    arr = np.zeros(len(un)-1)
    for i in range(1,len(un)):
        m = np.argmax(un[i])
        axarr.plot(ind[i],m,'.',color=colors[m])
        #if m != 7:
        #    arr[i-1]=m
        #    axarr[1].axvspan(i-0.5, i+0.5, facecolor=colors[m])
    axarr.set_title('Thematic Dispersion plot for Unique Phrases in Performance '+str(p))
    axarr.set_xlabel('Message Number')

def convert_array(p):
    p_temp = np.zeros((len(p),10))
    for i in range(len(p)):
        for j in range(len(p[i])):
            #print i,j
            p_temp[i,j]=p[i][j]
    return p_temp

path = '/Users/mb/Desktop/Janis.So/06_qmul/01_banda/banda_data/new_data'
p_1 = np.genfromtxt(path+'/p1.csv',delimiter=',',names=True)
p_2 = np.genfromtxt(path+'/p2.csv',delimiter=',',names=True)
p_3 = np.genfromtxt(path+'/p3.csv',delimiter=',',names=True)
p_4 = np.genfromtxt(path+'/p4.csv',delimiter=',',names=True)


all_tings = list(p_1)+list(p_2)+list(p_3)+list(p_4)





plt.figure()
ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4,rowspan=4)
ax2 = plt.subplot2grid((5, 4), (4, 0))
ax3 = plt.subplot2grid((5, 4), (4, 1))
ax4 = plt.subplot2grid((5, 4), (4, 2))
ax5 = plt.subplot2grid((5, 4), (4, 3))

#f, axarr = plt.subplots(2, 2)
e_all,order = big_pie(ax1,np.array(all_tings))
e_1 = small_pie(ax2,p_1,order,'P1')
e_2 = small_pie(ax3,p_2,order,'P2')
e_3 = small_pie(ax4,p_3,order,'P3')
e_4 = small_pie(ax5,p_4,order,'P4')
ax1.set_title('Thematic Occurance for all Performances')
plt.tight_layout()





#plt.figure()'''

ind_1 = np.array(p1_un)[:,0].astype(int)
ind_2 = np.array(p2_un)[:,0].astype(int)
ind_3 = np.array(p3_un)[:,0].astype(int)
ind_4 = np.array(p4_un)[:,0].astype(int)


p_1_temp = convert_array(p_1)
p_2_temp = convert_array(p_2)
p_3_temp = convert_array(p_3)
p_4_temp = convert_array(p_4)

theme_time_o(p_1_temp,ind_1,1,order)
theme_time_o(p_2_temp,ind_2,2,order)
theme_time_o(p_3_temp,ind_3,3,order)
theme_time_o(p_4_temp,ind_4,4,order)

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

'''labels = 'Phatic','Metalinguistic','Someone','Political','Onomatopaic','Comic','Statements','Abstract','Code','Emoticons'
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
ax.set_xticklabels(labels)
ax.legend((rects1[0],rects2[0],rects3[0],rects4[0]),('P1','P2','P3','P4'))
ax.set_title('Comparison of Recurrent Thematic Occurence Between Performances')



#f, axarr = plt.subplots()




np.flip(np.argsort(np.array([4,2,3,6,1])),axis=0)


N = 4

ab = (e_1[order][0],e_2[order][0],e_3[order][0],e_3[order][0])
st = (e_1[order][1],e_2[order][1],e_3[order][1],e_3[order][1])
ml = (e_1[order][2],e_2[order][2],e_3[order][2],e_3[order][2])
ph = (e_1[order][3],e_2[order][3],e_3[order][3],e_3[order][3])
so = (e_1[order][4],e_2[order][4],e_3[order][4],e_3[order][4])
on = (e_1[order][5],e_2[order][5],e_3[order][5],e_3[order][5])
co = (e_1[order][6],e_2[order][6],e_3[order][6],e_3[order][6])
ht = (e_1[order][7],e_2[order][7],e_3[order][7],e_3[order][7])
em = (e_1[order][8],e_2[order][8],e_3[order][8],e_3[order][8])
po = (e_1[order][9],e_2[order][9],e_3[order][9],e_3[order][9])

cumsums = np.zeros((10,4))
cumsums[0,:]=ab
cumsums[1,:]=cumsums[0,:]+st
cumsums[2,:]=cumsums[1,:]+ml
cumsums[3,:]=cumsums[2,:]+ph
cumsums[4,:]=cumsums[3,:]+so
cumsums[5,:]=cumsums[4,:]+on
cumsums[6,:]=cumsums[5,:]+co
cumsums[7,:]=cumsums[6,:]+ht
cumsums[8,:]=cumsums[7,:]+em
cumsums[9,:]=cumsums[8,:]+po

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

p1 = ax1.bar(ind, ab, width)
p2 = ax1.bar(ind, st, width, bottom=cumsums[0,:])
p3 = ax1.bar(ind, ml, width, bottom=cumsums[1,:])
p4 = ax1.bar(ind, ph, width, bottom=cumsums[2,:])
p5 = ax1.bar(ind, so, width, bottom=cumsums[3,:])
p6 = ax1.bar(ind, on, width, bottom=cumsums[4,:])
p7 = ax1.bar(ind, co, width, bottom=cumsums[5,:])
p8 = ax1.bar(ind, ht, width, bottom=cumsums[6,:])
p9 = ax1.bar(ind, em, width, bottom=cumsums[7,:])
p10 = ax1.bar(ind, po, width, bottom=cumsums[8,:])

#plt.ylabel('Ratio of Themes in Performances')
ax1.set_title('Ratio of Recurrent Themes in Performances')
#ax1.set_xticks(ind, ('P1', 'P2', 'P3', 'P4'))
ax1.set_xticks(np.arange(4))
ax1.set_xticklabels(('P1', 'P2', 'P3', 'P4'))
ax1.set_ylim(0,1)
#plt.yticks(np.arange(0, 81, 10))
#plt.legend((p1[0], p2[0], p3[0], p4[0],p5[0],p6[0],p7[0],p8[0],p9[0],p10[0]), list(np.array(labels)[order]))
plt.legend((p10[0], p9[0], p8[0], p7[0],p6[0],p5[0],p4[0],p3[0],p2[0],p1[0]), list( np.flip( np.array(labels)[order],axis=0)),bbox_to_anchor=(1,1),loc=2)
#plt.tight_layout()
#plt.show()'''
