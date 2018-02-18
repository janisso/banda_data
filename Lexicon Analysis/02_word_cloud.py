import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

path = '/Users/mb/Desktop/Janis.so/06_qmul/01_banda/Lexicon Analysis/thematic_analysis'
p_1 = np.genfromtxt(path+'/p1.csv',delimiter=',',names=True)


from wordcloud import WordCloud

# Read the whole text.
text = s

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt


# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(background_color="white",max_words=len(s),max_font_size=40, relative_scaling=.5).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()