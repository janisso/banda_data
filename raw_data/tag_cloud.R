library(janeaustenr)
library(tm)
library(SnowballC)
library(wordcloud)

mydata <- paste("/Users/mb/Desktop/Janis.so/06_qmul/01_banda/raw_data/p1no_number.txt", collapse = ' ',stopwords('en'))

#wordcloud(word_count$word[1:50], word_count$count[1:50])

#mydata = read.csv("/Users/mb/Desktop/Janis.so/06_qmul/01_banda/raw_data/p1_raw_un.txt")
wordcloud(mydata, max.words = 100, random.order = FALSE)
