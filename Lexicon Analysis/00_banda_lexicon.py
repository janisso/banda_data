from __future__ import division  # Python 2 users only

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

import nltk, re, pprint
from nltk import word_tokenize

import seaborn as sns
sns.set()
sns.set_style("whitegrid")
flatui = ["#0099ff", "#ff00cc", "#ff9900", "#ff0000","#95a5a6"]
sns.color_palette(flatui)
sns.set_palette(flatui)
sns.palplot(sns.color_palette(flatui))

#from pytagcloud import create_tag_image, make_tags
#from pytagcloud.lang.counter import get_tag_counts

#plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
##Options
#params = {'text.usetex' : True,
#          'font.size' : 11,
#          'font.family' : 'lmodern',
#          'text.latex.unicode': True,
#          }
#plt.rcParams.update(params) 

#THIS FUNCTION RETURNS NUMBER OF LINES IN THE FILE
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_letter_freq(array):
    text = ''
    for i in range(len(array)):
        text+=array[i][1]
    return text

#def unique_participants(array):
#    u_p = []
#    for i in range(len(array)):
#        text = array[i][1]
#        if text.startswith("[:open"):
#            u_p.append(text)
#    return u_p

#def unique_user_plot(array):
    
def get_data(logs_array):
    users = []
    user_count = 0
    unique = [] #UNIQUE MESSAGES
    non_unique = [] #NON-UNIQUE MESSAGES
    u_o = []
    u_c = []
    beg = 24
    end = 18
    str1 = '"touch\\":0}"]\n'
    str2 = '"touch\\":1}"]\n'
    samples = []
    for i in range(len(logs_array)):
        text = logs_array[i][1] 
        if text.startswith("[:open"):
            user_count+=1
            #u_p.append(text)
            #print text[8:-2]
            #print 'open ',text.split(',')
            split_array = text.split(',')
            if len(split_array) == 2:
                #print split_array[1][:-2]
                u_o.append([i,int(split_array[1][:-2])])
            else:
                u_o.append([i,int(split_array[1])])
        if text.startswith("[:close"):
            user_count-=1
            #print text.split(',')[1]
            split_array = text.split(',')
            if len(split_array) == 2:
                #print split_array[1][:-2]
                u_c.append([i,int(split_array[1][:-2])])
            else:
                u_c.append([i,int(split_array[1])])#c.append(text.split(',')[1])
            #split_array = text.split(',')
            #print 'close ',text.split(',')
        if text.endswith(str1):
            new_str = text[beg:-end]
            if new_str.startswith('/samples') == False:
                unique.append([i,new_str])
            if new_str.startswith('/samples') == True:
                if new_str.endswith('3'):
                    samples.append([i,3])
                elif new_str.endswith('2'):
                    samples.append([i,2])
                elif new_str.endswith('2'):
                    samples.append([i,2])
                else:
                    samples.append([i,0])
        if text.endswith(str2):
            new_str = text[beg:-end]
            non_unique.append([i,new_str])
                #print new_str
        #print text[-14:]==str
        users.append([i,user_count])
    return np.array(users),unique,non_unique,np.array(u_o),np.array(u_c),np.array(samples)

def plot_users(p,u,uo,uc,sa,un,non,l,t):
    #f,axarr = plt.subplots(1,sharex=True)
    #plt.figure(figsize=(8,5))
    axarr.plot(t/p[:,0],p[:,1],label=l)#,alpha=0.2)
    plot_timeline(axarr,p,uo,uc,sa,un,non)
    
def plot_timeline(axarr,p,uo,uc,sa,un,non):
    maxim = max(p[:,1])
    uo_array = np.zeros((maxim,len(p)))
    #uo_array.shape
    print uo_array.shape
    #for i in range(0,len(uo)):#len(uo)):
    for i in range(0,len(uo)):
        #print i
        uc_ind = np.where(uo[i,1]==uc[:,1])[0]
        #print p1_uc_ind
        if len(uc_ind)==1:
            #print 'Open ',uo[i,0],' Close ',uc[uc_ind,0][0]
            s = uo[i,0]
            e = uc[uc_ind,0][0]
            #axarr.plot([s,e],[p[s,1],p[s,1]])
            arry = np.zeros((e-s,2)).astype(int)
            arry[:,0]=np.arange(s,e)
            arry[:,1]=np.full(len(arry),p[s,1])
            #print arry
            #print arry
            #print arry
            for k in range(0,len(arry)):
                #print k
                x = arry[k,0]
                #y = arry[k,1]
                for j in range(0,arry[0,1]):
                    #print j,x
                    if uo_array[j,x]==0:
                        #print 'here'
                        uo_array[j,x]=i+1
                        arry[k,1]=j+1
                        break
                #print x,y
                #if uo_array[x,y-1] == 0:
                #    uo_array[x,y-1] = 1
                #    arry[k,1]=y-1
                #s = arry[k,1]
                #print s
            #print maxim, i, arry
            #for k in range(0,len())
            #print uo_array[:,i]
            #print arry
            #for j in range(len(arry)):
            #    found = 0
            #    yo_array[]
                #if uo_array[arry[j,0] ,arry[j,1]-1]==0:
                #    arry[j,1] = arry[j,1]-1
                #    uo_array[arry[j,0],arry[j,1]-1] = 1
            x = arry[:,0]
            y = arry[:,1]
            #interp_f = interp1d(x,y,kind='slinear')
            #x_new = np.linspace(x[0],x[-1],len(x)*5)
            #y_new = interp_f(x_new)
            axarr.plot(x,y,color='k',alpha=0.5,lw=2)
            axarr.plot(x[0],y[0],'.',color='g')
            axarr.plot(x[-1],y[-1],'.',color='r')
    for i in range(len(sa)):
        if sa[i,1]==0:
            c = 'blue'
        elif sa[i,1]==1:
            c = 'green'
            print 'green'
        elif sa[i,1]==2:
            c = 'red'
        else:
            c = 'm'
        #axarr.axvline(x=sa[i,0],color=c,alpha=0.2)
        axarr.plot(sa[i,0],0,'x',color=c,alpha=0.2)
    x = np.array(un)[:,0].astype(int)
    y = np.full(len(x),-1)
    axarr.plot(x,y,'o',color='green',alpha=0.3)
    
    x = np.array(non)[:,0].astype(int)
    y = np.full(len(non),-2)
    axarr.plot(x,y,'o',color='red',alpha=0.3)
    #for i in range(len(un)):
    #    axarr.plot(un[i][0],-1,'o',color='green',alpha=0.3)
    #for i in range(len(non)):
    #    axarr.plot(non[i][0],-2,'o',color='red',alpha=0.3)
    
def tokenise(un,l=0):
    raw = ''
    for i in range(0,len(un)):
        if l==1:
            raw+=un[i][1].lower()
        else:
            raw+=un[i][1]
        raw+=' '
    return raw
    #for i in range():
    #    
    #plt.imshow(uo_array,interpolation='nearest', aspect='auto')
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    #X,Y = np.meshgrid(np.arange(len(p)),np.arange(maxim))
#    X,Y = np.mgrid[0:len(p), 0:maxim]
#    print X.shape
#    print Y.shape
#    print uo_array.shape
#    #X,Y = np.mgrid[
#    #Y = np.meshgrid(np.arange(maxim))
#    ax.plot_surface(X.T, Y.T, uo_array,cmap=plt.cm.gray)
    #plt.figure()
    #plt.surf(uo_array)
    #    
    #plt.xlabel('Message Number')
    #plt.ylabel('User Count')
    #plt.suptitle('Performance '+str(u))

def save_csv(name,array,r):
    path = '/Volumes/KINGSTON/raw_data/'
    file = open(path+name+r+".txt","w")
    for j in range(0,len(array)):
        file.write('%s\n'%(array[j]))
    file.close()
    
    
    
    
import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]


#HERE WE COUNT THE NUMBER OF USERS
line_count = file_len('/Users/mb/Desktop/Janis.so/06_qmul/01_banda/banda/puma.out')

#ARRAY WITH INDEXES AND TEXT
#log_array = np.zeros((line_count,2)).astype(str) 
log_array = []

f = open('/Users/mb/Desktop/Janis.so/06_qmul/01_banda/banda/puma.out')
log_count = 0

for line in f:
    log_array.append([log_count,line])
    log_count+=1

p1 = [12,6951]
p2 = [6952,9497]
p3 = [9498,11920]
p4 = [11921,line_count]

p1_us, p1_un, p1_non, p1_uo, p1_uc, p1_s = get_data(log_array[p1[0]:p1[1]])
p2_us, p2_un, p2_non, p2_uo, p2_uc, p2_s = get_data(log_array[p2[0]:p2[1]])
p3_us, p3_un, p3_non, p3_uo, p3_uc, p3_s = get_data(log_array[p3[0]:p3[1]])
p4_us, p4_un, p4_non, p4_uo, p4_uc, p4_s = get_data(log_array[p4[0]:p4[1]])

print 'Unique Open Perf 1 ', len(set(p1_uo[:,1]))
print 'Unique close Perf 1 ', len(set(p1_uc[:,1]))
print 'Unique Logins Perf 2 ', len(set(p2_uc[:,1]))
print 'Unique close Perf 2 ', len(set(p2_uc[:,1]))
print 'Unique Logins Perf 3 ', len(set(p3_uc[:,1]))
print 'Unique close Perf 3 ', len(set(p3_uc[:,1]))
print 'Unique Logins Perf 4 ', len(set(p4_uc[:,1]))
print 'Unique close Perf 4 ', len(set(p4_uc[:,1]))

#plt.figure()
#for i in range(len(p1_uc)):
#    p1_uc_ind = np.where(p1_uc[i,1]==p1_uo[:,1])[0]
#    #print p1_uc_ind
#    if len(p1_uc_ind)==1:
#        print 'Open ',p1_uo[i,0],' Close ',p1_uc[p1_uc_ind,0][0]
#        s = p1_uo[i,0]aaawh
#        e = p1_uc[p1_uc_ind,0][0]
#        plt.plot([s,e],[i,i])
#save_csv('p1',p1_un)
#save_csv('p2',p2_un)
#save_csv('p3',p3_un)
#save_csv('p4',p4_un)
#
##text = get_letter_freq(p4_un)
#
##str2 = '$'
#path = '/Volumes/KINGSTON/raw_data/'
#file = open(path+'testfile.txt','w') 
# 
#file.write('Hello World') 
#file.write('This is our new text file') 
#file.write('and this is another line.') 
#file.write('Why? Because we can.') 
# 
#file.close()

#plot_users(p1_us,1,p1_uo, p1_uc)
#plot_users(p2_us,2,p2_uo, p2_uc)
f,axarr = plt.subplots(1,sharex=True)
plot_users(p1_us,1,p1_uo, p1_uc,p1_s,p1_un, p1_non,'P1',20.)
plot_users(p2_us,2,p2_uo, p2_uc,p2_s,p2_un, p2_non,'P2',15.)
plot_users(p3_us,3,p3_uo, p3_uc,p3_s,p3_un, p3_non,'P3',20.)
plot_users(p4_us,4,p4_uo, p4_uc,p4_s,p4_un, p4_non,'P4',20.)
axarr.set_xlabel('Time in Minutes')
axarr.set_ylabel('Number of Simultaneous Users')
axarr.legend()
#plot_users(p4_us,4)

#from nltk.corpus import words
#corp1 = list(set(nltk.corpus.mac_morpho.words()))

corp = []
f = open('por_bra.dic', 'r')
for line in f:
    if len(line.split('/'))==2:
        corp.append(line.split('/')[0].lower())
corp = list(set(corp))

#corp = []
#
#for i in range(len(corp1)):
#    corp.append(corp1[i].lower())

raw_1 = tokenise(p1_un,0)#.encode('utf-8')
token_1 = raw_1.split(' ')
#print len(token_1)
#token_1 = list(set(token_1))
#print len(token_1)
const = ['a','e','o']
real_words = []
non_real_words = []
#
'''for i in range(len(token_1)):
    print i
    if len(token_1[i])>1 or (token_1[i] in const):
        if token_1[i] in corp:
            print 'real word'
            real_words.append(token_1[i])
        else:
            non_real_words.append(token_1[i])
    else:
        non_real_words.append(token_1[i])
print len(real_words)


p1_una = np.array(p1_un)
p1_una = p1_una[:,1]'''



#save_csv('p4',real_words,'real')
#save_csv('p4',non_real_words,'non_real')

########### MESSAGE LENGTH

#YOUR_TEXT = "A tag cloud is a visual representation for text data, typically\
#used to depict keyword metadata on websites, or to visualize free form text."

#tags = make_tags(get_tag_counts(raw_1), maxsize=120)

#create_tag_image(tags, 'cloud_large.png', size=(900, 600), fontname='Lobster')



#users = np.array(users)
#
#plt.plot(users[:,0],users[:,1])
#plt.suptitle('User Count')
#plt.xlabel('Message Number')
#plt.ylabel('Number Users')

#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#
## generate some sample data
#import scipy.misc
#lena = scipy.misc.ascent()
#
## downscaling has a "smoothing" effect
#lena = scipy.misc.imresize(lena, 0.15, interp='cubic')[:10]
#
## create the x and y coordinate arrays (here we just use pixel indices)
#xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]
#
## create the figure
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.gray,
#        linewidth=0)
#
## show it
#plt.show()

#text = token_1
#
#
#
#from sklearn.feature_extraction.text import CountVectorizer
#from PIL import Image, ImageDraw, ImageFont
#
##cv = CountVectorizer(min_df=0, charset_error="ignore", stop_words=None, max_features=200)
#cv = CountVectorizer()
#
#
#counts = cv.fit_transform(list(text))#.toarray().ravel()                                                  
#words = np.array(cv.get_feature_names()) 
## normalize                                                                                                                                             
#counts = counts / float(counts.max())
#
#
#img_grey = Image.new("L", (1024, 768))
#draw = ImageDraw.Draw(img_grey)
#
#font = ImageFont.truetype('/Users/mb/Downloads/helvetica/HELR45W.ttf', 15)
#draw.setfont(font)
#draw.text((y, x), "Text that will appear in white", fill="white")