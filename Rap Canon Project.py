import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.text import Text
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
plt.style.use('ggplot')

#importing main dataset
df = pd.read_csv('/Users/erikrice/Downloads/Rap Canon Project - Sheet1.csv')
print(df.head())

#let's start by looking at geography. where are the most rappers on the top 50 list from?
print(df['City'].value_counts())

#prortionally?
print(df['City'].value_counts(normalize=True))

#new york is absolutely dominant. let's get a couple visuals
sns.set_context("paper", font_scale=0.95)  
sns.catplot(kind='count', x='City', data=df)
plt.xticks(rotation=90)
plt.xlabel('City')
plt.ylabel('# of Rappers')
plt.title('Cities Top 50 Rappers Are From or Most Associated With', y=.95)
plt.show()

#what about a more proportional visual of new york's dominence? 
ny_pie = np.array([46, 54])
ny_pie_labels = ['New York', 'All Other Cities']
ny_pie_colors = ['#6495ED', '#FFF8DC']
plt.pie(ny_pie, labels=ny_pie_labels, autopct='%1.1f%%', colors=ny_pie_colors)
plt.title('Pie Chart: Where Are The Top 50 Rappers From?')
plt.show()

#for reference, let's compare this to NYC's population to the rest of the country. (and this is exclusing the rest of the world!)
ny_pie2 = np.array([8.47, 323.43])
ny_pie_labels2 = ['New York', 'Rest of US']
plt.pie(ny_pie2, labels=ny_pie_labels2, autopct='%1.1f%%', colors=ny_pie_colors)
plt.title('Population of NYC Compared to Rest of US')
plt.show()

#what about the top of the list? creating a couple variables
top25 = df[df['Rank'] < 26]
top10 = df[df['Rank'] < 11]

#is the story different at the top of the list?
print(top25['City'].value_counts(normalize=True))
print(top10['City'].value_counts(normalize=True))

#let's see a side-by-side comparison
New_York = [46, 52, 40]
All_Other_Cities = [54, 48, 60]
n=3
r = np.arange(n)
width = 0.25
plt.bar(r, New_York, color = '#B0E0E6',
        width = width, edgecolor = 'black',
        label='New York')
plt.bar(r + width, All_Other_Cities, color = '#CD853F',
        width = width, edgecolor = 'black',
        label='All Other Cities')
plt.xlabel("Inclusion")
plt.ylabel("% Rappers")
plt.title("Percent of Rappers from New York vs. All Other Cities")
plt.xticks(r + width/2,['Top 50','Top 25','Top 10'])
plt.legend()
plt.show()

#broadening things out to region
print(df['Region'].value_counts())

#prortionally?
print(df['Region'].value_counts(normalize=True))

#let's get a couple visuals for this regional data
sns.set_context("paper", font_scale=0.9)  
sns.catplot(kind='count', x='Region', data=df)
plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('# of Rappers')
plt.title('Region Top 50 Rappers Are From (or Most Associated With)', y=.95)
plt.show()

#unsurprisingly now, the east is quite dominant. 
region_pie = np.array([58, 14, 20, 6, 2])
region_labels = ['East Coast', 'West Coast', 'South', 'Midwest', 'Canada']
region_colors = ['#A52A2A', '#7FFF00', '#6495ED', '#00FFFF', '#B8860B']
plt.pie(region_pie, labels=region_labels, autopct='%1.1f%%', colors=region_colors)
plt.title('Pie Chart: Where Are The Top 50 Rappers From?')
plt.show()

#what about the top of the list?
print(top25['Region'].value_counts(normalize=True))
print(top10['Region'].value_counts(normalize=True))

#let's see a side-by-side comparison for the regional data
East_Coast = [58, 56, 40]
All_Other_Regions = [42, 44, 60]
n=3
r = np.arange(n)
width = 0.25
plt.bar(r, East_Coast, color = '#E9967A',
        width = width, edgecolor = 'black',
        label='East Coast')
plt.bar(r + width, All_Other_Regions, color = '#2F4F4F',
        width = width, edgecolor = 'black',
        label='All Other Regions')
plt.xlabel("Inclusion")
plt.ylabel("% Rappers")
plt.title("Percent of Rappers from East Coast vs. All Other Regions")
plt.xticks(r + width/2,['Top 50','Top 25','Top 10'])
plt.legend()
plt.show()

#I made a higher quality stacked bar chart of this regional comparison on another program. it can be seen on my homepage and portfolio

#what about the living/deceased variable?
print(df['Living/Deceased'].value_counts())
print(df['Living/Deceased'].value_counts(normalize=True))

#over 90% of rappers on the list are still with us. possibly belies the sometimes popular notion that we overrate dead artists
#let's see a quick visual
living_pie = np.array([92, 8])
living_labels = ['Living', 'Deceased']
living_colors = ['#B22222', '#696969']
plt.pie(living_pie, labels=living_labels, autopct='%1.1f%%', colors=living_colors)
plt.title('Pie Chart: Are More Top 50 Rappers Living or Deceased')
plt.show()

#what about gender? how male-dominated is the canon?
print(df['Gender'].value_counts())
print(df['Gender'].value_counts(normalize=True))

#as expected, almost 90% of list is male. let's see a quick visual
gender_pie = np.array([88, 12])
gender_labels = ['Male', 'Female']
gender_colors = ['#00BFFF', '#FF1493']
plt.pie(gender_pie, labels=gender_labels, autopct='%1.1f%%', colors=gender_colors)
plt.title('Pie Chart: How Male-Dominated is the Hip Hop Canon?')
plt.show()

#what about age? let's have a look by decade. creating a column for this
birth_decade = []
for x in df['Birth Year']:
    if x < 1960:
        birth_decade.append('50s')
    elif x < 1970:
        birth_decade.append('60s')
    elif x < 1980:
        birth_decade.append('70s')
    else:
        birth_decade.append('80s')
df['Birth Decade'] = birth_decade
print(df.head())
      
#let's get a visual of the new decade column
sns.countplot(data=df, x='Birth Decade', order=df['Birth Decade'].value_counts().index)
plt.xlabel('Birth Decade')
plt.ylabel('# of Appearances on List')
plt.title('Top 50 Rapper List: Decade of Birth')
plt.show()

#the 70's dominate. let's look at proportionally
print(df['Birth Decade'].value_counts(normalize=True))
birth_pie = np.array([4, 22, 58, 16])
birth_labels = ['50s', '60s', '70s', '80s']
birth_colors = ['#A52A2A', '#7FFF00', '#6495ED', '#00FFFF']
plt.pie(birth_pie, labels=birth_labels, autopct='%1.1f%%', colors=birth_colors)
plt.title('Pie Chart: What Decade Were The Top 50 Rappers Born In?')
plt.show()

#out of curiosity, how old are these rappers right now?
df['Current Age'] = 2023 - df['Birth Year']
sns.catplot(data=df, kind='count', x='Current Age')
plt.title('How Old Is The Rap Cannon Right Now?', y=.95)
plt.show()

#early 50's is by far the most common age for rappers on this list. 
#what about a correlation between birth year and rank?
birth_year_corr = df['Birth Year'].corr(df['Rank'])
print(birth_year_corr)

#let's get a visual of this
fig, ax = plt.subplots()
ax.scatter(df['Birth Year'], df['Rank'])
ax.invert_yaxis()
plt.xlabel('Birth Year')
plt.ylabel('List Rank')
plt.title('Billboards Top 50 Rappers: Does Birth Year Dictate List Placement?', y=.95)
plt.show()

#let's look at this scatterplot with a couple other variables
sns.set_palette('Set2')
region_scatter = sns.scatterplot(data=df, x='Birth Year', y='Rank', hue='Region')
region_scatter.invert_yaxis()
region_scatter.figure
region_scatter.set_xlabel('Birth Year')
region_scatter.set_ylabel('List Ranking')
region_scatter.set_title('Top 50 Rappers Birth Year Compared to List Placement and Region')
plt.show()

#what about the living/deceased variable?
sns.set_palette('Dark2')
living_scatter = sns.scatterplot(data=df, x='Birth Year', y='Rank', hue='Living/Deceased')
living_scatter.invert_yaxis()
living_scatter.figure
living_scatter.set_xlabel('Birth Year')
living_scatter.set_ylabel('List Ranking')
living_scatter.set_title('Top 50 Rappers Birth Year Compared to List Placement and Living/Deceased Status')
plt.show()

#what about gender?
sns.set_palette('Accent')
gender_scatter = sns.scatterplot(data=df, x='Birth Year', y='Rank', hue='Gender')
gender_scatter.invert_yaxis()
gender_scatter.figure
gender_scatter.set_xlabel('Birth Year')
gender_scatter.set_ylabel('List Ranking')
gender_scatter.set_title('Top 50 Rappers Birth Year Compared to List Placement and Gender')
plt.show()

#let's turn to the apex data. when did the rap canon peak?
peak_decade = []
for x in df['Biggest Year']:
    if x < 1990:
        peak_decade.append('80s')
    elif x < 2000:
        peak_decade.append('90s')
    elif x < 2010:
        peak_decade.append('00s')
    else:
        peak_decade.append('10s')
df['Peak Decade'] = peak_decade
print(df)

#let's take a look at the "peak" decade data
sns.set_palette('PuOr')
sns.catplot(data=df, kind='count', x='Peak Decade', order=df['Peak Decade'].value_counts().index)
plt.title('What Decade Did Most Top 50 Rappers Experience Their Peak?', y=.95)
plt.show()

#how old were most rappers during their peak?
df['Age At Peak'] = df['Biggest Year'] - df['Birth Year']
print(df.dtypes)

#visual of their age during their peak
sns.set_palette('RdGy')
sns.catplot(data=df, kind='count', x='Age At Peak')
plt.title('What Age Were Most Rappers At For Their Peak?', y=.95)
plt.show()

#is there any relationship to when a rapper had their commercial peak and their placement on this list?
peak_scatter = sns.scatterplot(data=df, x='Biggest Year', y='Rank', color='#696969')
peak_scatter.invert_yaxis()
peak_scatter.figure
peak_scatter.set_xlabel('Year of Commercial Peak')
peak_scatter.set_ylabel('List Ranking')
peak_scatter.set_title('What Year Did Billboard Top 50 Rappers Experience Their Commercial Peak?')
plt.show()

#what about the age they were at their peak and their placement on the list?
peak_age_scatter = sns.scatterplot(data=df, x='Age At Peak', y='Rank', color='#B22222')
peak_age_scatter.invert_yaxis()
peak_age_scatter.figure
peak_age_scatter.set_xlabel('Year of Commercial Peak')
peak_age_scatter.set_ylabel('List Ranking')
peak_age_scatter.set_title('Age of Billboards Top 50 Rappers At Commercial Peak')
plt.show()

#did different regions peak at different times? let's look at a cross section of that data
sns.set_palette('Dark2')
sns.relplot(data=df, x='Biggest Year', y='Rank', hue='Region')
plt.xlabel('Year of Commercial Peak')
plt.ylabel('List Ranking')
plt.suptitle('Did Different Hip Hop Regions Peak At Different Times? (Source: Billboard)', y=1)
plt.ylim(reversed(plt.ylim()))
plt.show()

#what about gender? did the women of the canon peak during any particular time?
sns.set_palette('Pastel2')
sns.relplot(data=df, x='Biggest Year', y='Rank', hue='Gender')
plt.xlabel('Year of Commercial Peak')
plt.ylabel('List Ranking')
plt.suptitle('Did Women Rappers Peak At Any Particular Time? (Source: Billboard)', y=1)
plt.ylim(reversed(plt.ylim()))
plt.show()

#final indicator. what about how many #1 albums a rapper has? let's look at the count
sns.set_palette('Paired')
sns.catplot(kind='bar', x='Rapper', y='#1 Albums', data=df)
plt.xlabel('Rapper')
plt.ylabel('#1 Albums')
plt.suptitle('How Many #1 Albums Do Billboards Top 50 Rappers Have?')
plt.xticks(rotation=90)
plt.show()

#any correlation between this commercial success and list placement?
sns.relplot(kind='scatter', x='#1 Albums', y='Rank', data=df)
plt.xlabel('#1 Albums')
plt.ylabel('List Ranking')
plt.suptitle('#1 Albums Compared to Billboard List Placement')
plt.ylim(reversed(plt.ylim()))
plt.show()

#let's look at some summary statistics before moving on to more advanced analysis
print(df.dtypes)
df_numeric = df[['Rank', 'Birth Year', 'Biggest Year', '#1 Albums', 'Current Age', 'Age At Peak']]
print(df_numeric.mean())
print(df_numeric.corr())

#heatmap for all numeric indicators
sns.heatmap(df_numeric.corr(), annot=True, cmap="BuPu")
plt.show()

#looking at a cluster analysis for the entire dataset
X0 = df_numeric.copy()
wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state=0)
    kmeans.fit(X0)
    wcss.append(kmeans.inertia_)
sns.set()
plt.plot(range(1, 11), wcss)
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

#elbow curve suggests the canon could be split into 4 clusters. could delve deeper, but we already split this data into myriad groups
#looking at some natural language processing now. using the first entrant, Rick Ross, as a test.
Rick_Ross = "Rozay’s “Hustlin’” dreams turned to gold when he rattled the cages of rap’s mainstream with his 2006 summer anthem and pledged his allegiance to Def Jam and its ex-president Jay-Z. Ross demanded attention through his deep gruff-and-grunting ad-libs, colorful street tales and inside looks at Miami’s lavish lifestyle, plus his incredible ear for production — more appetizing than a lemon-pepper Wingstop order. His catalog reigns supreme, especially in the late 2000s and 2010s, as he doled out gems such as Trilla, Deeper Than Rap and God Forgives, I Don’t. Ross has notched 58 Billboard Hot 100 entries spanning three decades, but his legacy doesn’t end there: His Maybach Music Group laid the tarmac for Meek Mill and Wale to take off and become titans in the 2010s."
Ross_Tokens = nltk.word_tokenize(Rick_Ross)

#let's get rid of some stop words
stop_words_test = set(stopwords.words('english'))
filtered_list_test = []
for word in Ross_Tokens:
    if word.casefold() not in stop_words_test:
         filtered_list_test.append(word)
Ross_frequency_distribution = FreqDist(filtered_list_test)
print(Ross_frequency_distribution.most_common(20))

#looks good enough for testing purposes. let's expand this to the whole canon now.
#going to make two sets - one for the top ten and one for the top fifty for comparison
canon_top10 = "“I will not lose.” Brooklyn’s Shawn “Jay-Z” Carter’s defiant-yet-confident declaration has proven true through his legendary career. History is also on his side: Hov has 14 Billboard 200 No. 1 albums (the most amongst solo acts), and over 140 million records sold. He co-founded Roc-A-Fella Records, collected 24 Grammys, ran hip-hop’s legendary Def Jam Records as president, guided Rihanna and Ye to billionaire status, founded a behemoth entertainment/sports agency in Roc Nation and best-selling spirits brands D’usse and Armand de Brignac… and even today, still makes time to drop four-minute long rap verses that break the internet (as he did on DJ Khaled’s “God Did” in 2022). Beyonce Knowles-Carter’s husband has succeeded in every era, spitting regal-scented rhymes that speak to the struggle, and to the opulence that follows an intensely self-made greatness. The first rapper inducted into the Songwriters Hall of Fame, Jay-Z believes everyone has genius-level talent. Lucky for us, Jay found his early in life, and has since become your favorite rapper’s favorite rapper.  Maybe there are other rappers who could claim to match Jay at his peak. But there are simply none who can match his entire career — his longevity, the breadth of his accomplishments, and what he’s meant to hip-hop from a musical, cultural and financial standpoint for the last three decades. The Compton native has become one of music’s most influential artists thanks to his vivid, thought-provoking — and sometimes controversial — lyrics, fearless genre experimentation, and agile-yet-masterful flow. Lamar first gained local attention as teen rapper K.Dot through various mixtapes before releasing his first studio album on indie label Top Dawg Entertainment, 2011’s Section.80. After signing with Dr. Dre’s Interscope imprint Aftermath Entertainment a year later, Lamar hit the ground running with second studio album good kid, m.A.A.d City — becoming an instant creative and cultural force. Signaling a major renaissance in West Coast and gangsta rap, his critically acclaimed sophomore project reeled off several commercial hits: “Swimming Pool” (Drank), “Backseat Freestyle,” and “Bitch, Don’t Kill My Vibe.” Next came 2015’s jazz-influenced To Pimp a Butterfly, his first Billboard 200 No. 1. The evolution continued with the R&B, psychedelic soul, and pop-infused DAMN. Featuring Lamar’s first solo No. 1 single, “HUMBLE,” the album won the 2018 Pulitzer Prize for Music, the first non-jazz or classical work to do so. That year also marked his major foray into film with the Black Panther the Album. Last year brought Lamar’s final TDE album and latest addition to his conscious rap repertoire, Mr. Morale & the Big Steppers. Now the 17-time Grammy winner (Mr. Morale was just newly crowned as best rap album) and Emmy winner for last year’s Super Bowl halftime show is focusing his entrepreneurial sights on pgLang, his film, TV and music collective that has already scored success with next-gen rapper Baby Keem. Meanwhile, his enduring combination of lyrical prowess, classic albums and envelope-pushing evolution continue to support the widely held consensus that he’s the best rapper of his generation. At this point, hip-hop purists know Nas’ path to greatness all too well. In 1991, Large Professor tossed him an alley-oop on Main Source’s posse cut  “Live at the Barbeque,” and Escobar delivered a rim-rocker of a performance. He was crowned the Chosen One from the jump, and had LeBron-like expectations to overcome. But like King James, he relished the pressure, releasing his seismic debut album Illmatic in 1994 — a quintessential hip-hop opus that transcended the genre and became the Holy Bible for every budding lyricist. To this day, the album remains an undisputed classic, and he continued evolving and staying relevant, scoring No. 1 albums well into the 2010s. By now, his vivid storytelling (“Black Girl Lost” & “I Gave You Power”) and precocious diction (“Nas Is Like” & “Made You Look”) are goated, and his resiliency has proven to be Hall of Fame-level – especially when he scored a TKO against Jay-Z and added a new verb to the hip-hop lexicon with his scathing comeback diss “Ether” in 2001. Even with those shiny accomplishments, it’s Nasir Jones’ longevity and adaptability that make him a one-of-a-kind specimen. Since partnering with Hit-Boy in 2020 for their first installment of their well-received King’s Disease series, Nas has released three more projects and nabbed his first-ever Grammy, proving his lyrical sword remains as sharp as ever. Poet, actor, activist, and rapper Tupac Amaru Shakur was the ultimate polymath in the ’90s. Once a tag-along member in Digital Underground, ‘Pac’s larger-than-life demeanor became too big to shelter. after his scene-stealing verse on their 1991 hit “Same Song.” Pac’s duality as a solo artist was remarkable: One minute, he was endearing, scripting empathetic classics like “Dear Mama” and “Keep Your Head Up;” the other, he was explosive, lacing up his war boots, ready to ride on his enemies (“Hit ‘Em Up” and “Hail Mary”). He wasn’t necessarily the best lyricist, nor did he possess the craziest flow. But in fact, his limitations were what made the 25-year-old wunderkind one of the greatest MCs to touch God’s green earth: He knew of his shortcomings, but relied on his voice and motivational messaging to resonate and crack through the toughest gangstas. Signing to Death Row upon his 1995 prison release formed hip-hop’s strongest triumvirate, with him, Dr. Dre, and a burgeoning Snoop Dogg leading the way. His most potent project came when he released his double-disc album All Eyez On Me the next year. His fourth studio album had the makings of a bonafide classic – with incredible singles (“How Do You Want it,” “California Love,” & “I Ain’t Mad at Cha”) and superior production – and it reigned for eight weeks at No. 1 on the Billboard 200. Despite his gaudy wins, the mercurial MC had his fair share of battles, most notably with Brooklyn titan The Notorious B.I.G, which turned into heartbreaker for both coasts. Their lyrical skirmish ended in tragedy when ‘Pac was shot and killed in a Las Vegas shooting in ‘96, rocking the entire music world. Even after his death, Pac’s legacy pushed forward – with seven posthumous albums (including three Billboard 200-toppers), induction into the Rock Hall of Fame, and even a spot in the Library of Congress’ National Recording Registry (for “Dear Mama”) – and despite his short life and career, Pac’s legacy continues to resonate today as much as any rapper. After coming up in freestyle rap battles, Eminem has continually wielded the mic as his weapon of choice and obliterated anyone who comes in his path – as seen in the countless feuds he’s engaged in over the years. With unapologetically controversial and macabre (yet frequently hilarious) bars and unparalleled rhyme schemes and syncopation, he’s relished playing the role of rap’s supervillain. Em also certifiably spits a mile a minute, breaking Guinness World Records like fastest rap in a hit single with “Godzilla,” where he rapped 225 words in a 30-second segment. With 15 Grammy awards, 10 No. 1 Billboard 200 albums and three RIAA-certified Diamond singles (“Lose Yourself,” “Love the Way You Lie” and “Not Afraid”), his unprecedented commercial success makes for one of the most noteworthy rags-to-riches tales in popular music. His award-winning 2002 biographical film 8 Mile even depicted the Detroit-bred MC’s real-life struggle to be accepted as a white rapper in hip-hop, a genre created and dominated by Black people. But with his peerless technical skills, larger-than-life personality and turn-of-the-century run of classic albums, Slim Shady has rightfully earned his spot in the upper echelon of GOAT rappers lists. Christopher “The Notorious B.I.G.” Wallace is the ultimate rap phenomenon. Starting with 1993’s riotous “Party and Bullshit,” the Brooklyn, New York kingpin later signed with Diddy’s Bad Boy Records. Building a reputation for delivering gritty tales in a laid-back style, accented by deep-toned, rumbling vocals and a signature dark sense of humor, Biggie went on to score 16 Hot 100 hits – including two No. 1s with “Mo’ Money Mo’ Problems” and “Hypnotize.” Big completed two classic solo albums, 1994’s Ready to Die and 1997’s double-disc Life After Death – the latter spending four weeks at No. 1 on the Billboard 200. His executive producer status grew as he created his Junior Mafia clique – writing and producing their 1995 Conspiracy album, then releasing JM breakout star Lil Kim’s Hard Core debut in 1996. Six months after the death of Tupac Shakur, Biggie was murdered in a drive-by shooting in Los Angeles in 1997. So we’ll never know what the then-24-year-old could have accomplished if he’d been allowed a career as long and as storied as his peers. But through only two studio albums that still resonate nearly 30 years later, Wallace proved that a charismatic big man could mix lyrical street rhymes with heart and humor — and permeate the mainstream with style. “Bring the crowd and I’m loud in living color/ It is Weezy F–kin’ Baby, got these rappers in my stomach,” Lil Wayne opens his 2005 deep cut, “Best Rapper Alive,” foreshadowing the years to come. Wielding words like swords, Wayne is one of the most masterful lyricists of our time, delivering mind-boggling verses for the last quarter century. The former honors student dropped out of school at 14 to focus on his music career, five years after entering the mentorship of Cash Money Records co-founder Birdman. Around that time, Wayne joined The Hot Boys with fellow rappers Juvenile, B.G., and Turk and they topped Billboard’s Top R&B/Hip-Hop Albums chart in 1999 with Guerilla Warfare. It would be the first of twelve chart-topping albums for Wayne, including his iconic Tha Carter series, boasting a plethora of platinum plaques in the process. With the first-week million-seller Tha Carter III and its smash hits “A Milli” and “Lollipop”  in 2008 – the latter of which was his first Hot 100 No. 1, topping the chart for three weeks – Wayne successfully exploded onto the mainstream, setting the stage for rap hopefuls and fellow Billboard all-time chart stars Drake and Nicki Minaj, both of whom Wayne helped develop into the global icons they are today by way of his Young Money/Cash Money record label. But despite the crossover success, Wayne continued to feed his Mixtape Weezy fanbase with cult-favorite series like Da Drought, No Ceilings and Sorry 4 the Wait, and fearlessly reinvented himself with a rap-rock sound via 2010 album Rebirth, which continues to inspire new-school rappers like Lil Uzi Vert and Playboi Carti. Lil Wayne’s cultural, sonic and lyrical influence will forever be embedded into the fabric of hip-hop, by way of his talent as both an artist and as an A&R. Drizzy Drake Rogers forecasted his superstar climb when he was Wheelchair Jimmy on the popular ’00s teenage show Degrassi: The Next Generation. Since his 2009 breakthrough mixtape classic, So Far Gone, Drake’s ability to swerve in and out of genres and mix of sugary crooning with spiky bars made him an untouchable force: Anything he’s graced with his Midas touch has been zapped into platinum and gold. With his impenetrable streak of commercial success – he’s currently the Hot 100’s all-time hits leader, with nearly 300 entries to his credit – he refuses to give his opposition a breather, consistently releasing projects every calendar year, including generation-defining sets like 2011’s Take Care and 2013’s Nothing Was The Same. Whether he’s crooning his aches and pains on timeless gems (“Marvin’s Room” and “Jaded”) or slashing down hapless MCs (“5 AM in Toronto and “Omerta”), Drake’s virtuosic skillset has made him one of the most gifted (and most accomplished artists to ever touch down in music. Snoop Doggy Dogg. Snoop Lion. Snoop Dogg. No matter the moniker, there’s no denying Calvin Broadus Jr.’s estimable impact as one of the founding fathers of West Coast and gangsta rap. Alongside mentor Dr. Dre, the lanky Long Beach rapper introduced his ultra-cool demeanor and laid-back flow as the guest on the former’s 1992 debut solo single “Deep Cover.” The Chronic, Dre’s multi-Platinum-certified G-Funk classic (influenced by Parliament-Funkadelic’s psychedelic sound) arrived later that same year, led by one of the pair’s signature anthems, “Nuthin’ But a ‘G’ Thang.” The project was the springboard that rocketed Snoop and G-Funk to the top of the Billboard 200 with his 1993 Death Row Records solo debut, Doggystyle. The Dre-produced set further solidified the West Coast’s status as a major player in the rap game and featured such enduring hits as “Gin and Juice” and “Who Am I? (What’s My Name?)”, among the catchiest singles in hip-hop history. More seminal, million-plus-selling albums followed, such as Tha Doggfather, Da Game Is to Be Sold, Not to Be Told; R&G (Rhythm and Gangsta): The Masterpiece and Tha Last Meal as Snoop moved on from the G-Funk era to make albums with No Limit, score crossover hits with The Neptunes and even become an early adopter of Auto-Tune. Further showcasing his versatility, the rapper detoured into reggae as Snoop Lion on 2013’s Reincarnated, before reclaiming his Snoop Dogg persona for 2018’s Bible of Love, which debuted at No. 1 on Top Gospel Albums. A serial entrepreneur and activist, Snoop Dogg brought his career full-circle by purchasing his alma mater Death Row in 2022. Hailing from South Jamaica, Queens by way of Trinidad and Tobago, Nicki Minaj earned her crown as the modern-day Queen of Rap with her fierce, braggadocious spirit. Her 2009 mixtape Beam Me Up Scotty established her as a lyrical powerhouse, shortly before she emerged as the First Lady of Young Money, solidifying the label’s ineffable trinity of chart-conquering rap beasts with Lil Wayne and Drake. Over a slew of genre-bending albums (Pink Friday, Pink Friday: Roman Reloaded, The Pinkprint) and killer guest verses (most famously on Kanye West’s “Monster”), Nicki has defended her title for over a decade, with incredibly animated flows and alter egos – from the soft-spoken, pink-haired Harajuku Barbie to the volatile Roman Zolanski with the British cockney accent. She’s undeniably blazed the trail for the next generation of female MCs, while at the same time, building a legacy whose impact is hardly limited to the hip-hop world: Nicki is one of only a dozen artists to have more than 100 Billboard Hot 100 entries, and earned the MTV Video Vanguard Award in 2022 for her provocative music videos. "

#grabbing the rest of the list now (in parts)
canon1 = "Starting in the early ’00s with production on hits for Jay-Z, Beanie Sigel and Talib Kweli, Kanye West moved in front of the mic in 2003 with his solo debut, “Through the Wire,” then scored his first Hot 100 No. 1 alongside Twista and Jamie Foxx on “Slow Jamz.” Then came a torrent of beloved albums (Graduation, 808s & Heartbreak and My Beautiful Dark Twisted Fantasy). While his ascension from producer to artist has been remarkable, West’s quest as a fashion and business mogul, especially with his Yeezy empire, makes him one of the genre’s most impactful pioneers. In the late 2010s, West (legally known as “Ye”) became one of the most divisive artists in the world due to his political views and goading (occasionally outright antagonistic) social comments. From his “Slavery is a choice” remarks in 2018 to his more recent reliance on antisemitic tropes (which began during a troubling global rise in hate crimes against Jewish people), Ye’s fall from grace, amid ongoing reported mental health issues, has sadly been as monumental as his artistic output. “The South got something to say,” André 3000 said in rebuttal to the New York crowd booing OutKast for their best new rap group win at the 1995 Source Awards. He had a point, and OutKast certainly played a key role in that. The duo’s 1994 debut, Southernplayalisticadillacmuzik, used gospel-influenced tracks (produced by Organized Noize) to introduce the many nuances and characters of Atlanta. The group notched three No. 1 Hot 100 hits, sold 25 million albums and account for rap’s most recent album of the year win at the Grammys. André emerged as the star of the group for his singular flair; sadly, the duo went their separate ways with fans still yearning for a solo album from him. Still, on the guest side, Andre’s stamina and lyrical fervor remains top tier, crushing features for Jeezy (“I Do”), DJ Unk (“Walk It Out”), and UGK “Int’l Players Anthem.” Overall, 3000’s eclecticism, eccentricity and mind-blowing rhymes prove you can be out of this world and still cut through. When you consider the evolution of hip-hop, Rakim is the source material for pioneering the use of internal and multisyllabic rhymes, penning intricate lyrics and shifting the overall use of simple flows to more complex deliveries. When he partnered with fellow Long Island native Eric B., they became the mighty DJ and MC combo Eric B. & Rakim, producing classic albums like 1987’s Paid in Full and 1988’s Follow the Leader with eternal singles like “Eric B. Is President,” “Paid in Full” and “Microphone Fiend.” Rakim’s influence is felt in all corners of hip-hop: A$AP Rocky, born Rakim Mayers, is named after him, DMX and Eminem are self-proclaimed fans, and rappers as big as 50 Cent (via The Game’s “Hate It or Love It”) and Lil Wayne (on Lloyd’s “Girls Around the World”) have paid in-song homage to his distinctive flow. LL Cool J signed with Def Jam Records as its first artist in 1984 and released “I Need a Beat” from his studio debut, Radio, produced entirely by Rick Rubin. During rap’s golden age, “I Need a Beat” opened the door to the b-boy style, with unmatched energy and aggressive lyricism that became a genre hallmark. The fledgling label’s first album to crack the top 50, Radio spent 38 weeks on the Billboard 200 and claimed platinum status in 1989. LL spent his entire career at Def Jam, becoming their jack-of-all-trades — a hitmaker for the ladies and a multi platform trailblazer for artists to cross over into such other fields as TV and philanthropy. LL’s list of achievements includes being the first rapper to be awarded the prestigious Kennedy Center Honor. In 2022, he relaunched the Rock the Bells Festival in Queens, which included performances from Rick Ross, The Diplomats, Ice Cube and more. A defining byproduct of rap’s blog era, J. Cole made a name for himself with celebrated mixtapes like 2007’s The Come Up, 2009’s The Warm Up and 2010’s Friday Night Lights. Cole’s story is well-known: Hov signed him to Roc Nation as its first artist in 2009 after he garnered attention with “Lights Please” and dropped some blistering guest verses (Wale’s “Beautiful Bliss,” Kanye West’s “Looking for Trouble”). Down to earth and humble, Cole possesses a passion that pulsates throughout his raps and beats. Raised in Fayetteville, N.C., he’s Southern to the core — but also reps Queens as a St. John’s graduate. He can turn the mundane into something sweet (“Foldin Clothes”) or act out in rebellion (“Fire Squad”). He showcased his genuine connection to fans by rapping over a YouTube producer’s “J. Cole Type Beat” on “procrastination (broke).” He also went platinum with all five of his 2010s albums – the last three (2014 Forest Hills Drive, 4 Your Eyez Only and KOD) with no features, of course. Name a subject covered by your favorite rapper, and Scarface probably addressed it first. Rhymes about experiencing a mental health crisis? There’s “Mind Playing Tricks on Me,” his 1991 classic hit as a member of Houston’s Geto Boys, or “I’m Dead,” from his debut album that same year, Mr. Scarface Is Back. Bars about investing and saving money? There’s “Safe” from his 2002 album The Fix. Between his Geto Boys tenure in the late ’80s and his illustrious solo career stretching through the 2010s, Scarface laid bare the pitfalls of street life by presenting a stoic, pragmatic approach to hustling that differed from the flashy images glamorized by other artists. And he did all of it as one of the South’s first real rap stars, showcasing lyrical complexity and emotional depth in a region that was once wholly dismissed — while also making classic cuts with legends from both coasts, like Jay-Z, Nas and 2Pac. After an attempt on his life left him riddled with nine bullets, Curtis Jackson III revved up a meteoric resurrection. His 2002 mixtape Guess Who’s Back? prompted his signing to Eminem’s Shady Records under Dr. Dre’s Aftermath Entertainment banner, and was followed by the Billboard 200 No. 1 debut of Get Rich or Die Tryin’ in 2003. The album served up unapologetic street rhymes, dark humor and some of the stickiest choruses that year, thanks to two Hot 100 No. 1s: “In da Club” and “21 Questions” with Nate Dogg. Reviving gangsta rap in the process, Get Rich also nabbed a stunning nine-times platinum RIAA certification. Before Power-ing up his TV empire, 50 Cent continued to top the charts with follow-ups like The Massacre and Curtis, while also making stars of his G-Unit label roster mates, including Lloyd Banks, Young Buck and The Game. Before Ice Cube, born O’Shea Jackson, became a Hollywood heavyweight and basketball league owner, he wrote brilliant and incisive verses for his N.W.A groupmates (Dr. Dre, Eazy-E, Arabian Prince, DJ Yella and MC Ren) and himself on Straight Outta Compton, the group’s classic 1988 debut. Doubling as one of gangsta rap’s foundational platforms, the album featured the controversial anthems “Fuck Tha Police” and “Gangsta Gangsta,” and was certified triple-platinum despite minimal radio or MTV promotion. But even after writing rhymes that triggered FBI attention and radio bans, Ice Cube launched an even more extraordinary solo run. Albums like AmeriKKKa’s Most Wanted and Death Certificate still stand as some of rap’s most provocative works – and when he wasn’t sharpening his pro-Black commentary against the systemic racism crippling the community, he was equally adept at balancing tongue-in-cheek insights with feel-good storytelling, as on his timeless 1993 hit “It Was a Good Day.” One of hip-hop’s most creative visionaries, Missy “Misdemeanor” Elliott began honing her skills as a rapper, singer, songwriter and producer as a member of the R&B/hip-hop collective Swing Mob in the early ‘90s with childhood friend/producer Timbaland. After collaborating on projects by Aaliyah and others, the pair focused their attention on Elliott’s solo career, beginning with her arresting 1997 debut, Supa Dupa Fly. While turning heads with hit singles (“The Rain,” “Hot Boyz,” “Get Ur Freak On,” “Work It”) and acclaimed albums (Da Real World, Miss E … So Addictive, Under Construction, This Is Not a Test), she crafted a uniquely futuristic, funky and totally unique style of hip-hop that found her rhyming, singing, scatting or doing whatever else the beat might spontaneously spark. Lyrically, she pushed hip-hop beyond its boundaries regarding female empowerment, with a fearlessness that also imbued her innovative and still-influential music videos. In 2020, Elliott ranked No. 5 on Billboard’s 100 Greatest Music Video Artists of All Time. Big Daddy Kane was an early rap star archetype thanks to his show-stealing performances in the late 1980s. A pioneer in crafting the fast-flowing, double-time rap style, he attracted the ladies with his convincing, authoritative machismo and head-turning fashion sense. He also rolled deep with the Juice Crew, Marley Marl’s Queens-based squad whose members included MC Shan, Biz Markie, Kool G Rap and Roxanne Shante. But even among those heavy-hitters, Kane more than held his weight: Tracks like “Ain’t No Half Steppin’,” “Set It Off” and “Smooth Operator” sound as spirited now as they did decades ago. And Kane’s skill set remains sharp, whether he’s tackling police brutality on his 2020 cut “Enough” or reeling off a fantastic guest cameo on mentee Busta Rhymes’ 2022 song “Slap.” With a ravenous rhyme style and the powerful recurring use of dogs as a lyrical motif, DMX had the most commanding presence rap has ever seen. His pair of classic 1998 albums, It’s Dark and Hell Is Hot and Flesh of My Flesh, Blood of My Blood, along with the 1999 follow-up …And Then There Was X, topped the charts through grim, unforgiving street tales, processing the pains of a traumatic childhood and seeking spirituality for solace. When he wasn’t baring his soul, DMX steadily proved he could hang with rap’s best: LL Cool J, Jay-Z, Nas, Busta Rhymes and others have all experienced X’s untameable aggression firsthand. Unfortunately, the same demons that bore his provocative art led to his unfortunate demise: he died of a cocaine-induced heart attack in 2021. With a collection of brilliant solo albums and one of the most unmistakable styles in hip-hop history, Ghostface Killah is arguably the most accomplished rapper within the Wu-Tang Clan, which already has its own stake on any GOAT rap group list. The Staten Island lyricist draws from a Technicolor palette of inimitable slang, colorful character profiles, luxurious wares and slippery flows to craft vibrant, imaginative stories. He struck special chemistry with Wu compatriot Raekwon when the two adopted prominent roles on each other’s classic ‘90s debuts, Ironman and Only Built 4 Cuban Linx, respectively. Ghost continued the winning streak well into the 21st century, first with his 2000 sophomore opus Supreme Clientele and on through 2006’s brilliant Fishscale and 2013’s comic book on wax, Twelve Reasons to Die. Before rap became a cultural and commercial powerhouse, Kurtis Blow provided early proof of the genre’s future capabilities. He was the first rapper signed to a major label, created one of hip-hop’s earliest hits with “Christmas Rappin’,” earned the first gold plaque for a rap single with “The Breaks” and became the first rapper to embark on a national (and then international) tour. Blow may not possess his descendants’ complex rhyme skills, but he was a star in his own right thanks to a magnetic voice and charismatic stage presence. Plus, his songs were both fun and relatable: “The Breaks” cleverly recounted bad days; “Christmas Rappin’” still rings true as a holiday jam; and “If I Ruled the World” showcased rap’s early aspirations. His impact is undeniable: Nas, Jay-Z, De La Soul, KRS-One and Redman are just a few of the rappers who have sampled Blow’s music. KRS-One evolved from being The Blastmaster to The Teacha — a testament to his own lyrical and personal development. And if you ask him, it’s also a testament to the power of hip-hop. As a lyrical purveyor of the violence that littered his Bronx neighborhood, KRS-One started his career in the late ‘80s on Boogie Down Productions’ classic Criminal Minded. However, after the murder of the group’s DJ Scott La Rock, he shifted to socially conscious rap aimed at empowering Black people and resolving the ills plaguing inner-city communities. Works like BDP’s By All Means Necessary and the KRS-formed Stop the Violence Movement’s star-studded 1989 single “Self-Destruction” are some of the most powerful moments that hip-hop has ever witnessed. KRS-One is also battle-tested: 1987’s “The Bridge Is Over,” recorded in response to MC Shan’s “The Bridge,” is one of the most respected (and referenced) diss tracks of all time. Method Man is hip-hop’s version of create-a-player perfection. The Wu-Tang Clan member possesses it all: unmistakable voice, impenetrable flow, witty rhymes, dark sense of humor, charm and good looks that also translated into Hollywood success. Meth’s total package of street-savvy raps and heartthrob sex appeal made him a dominant force in the ‘90s and 2000s: he could carry his weight with his Wu brothers, spit muscular rhymes alongside Notorious B.I.G. and 2Pac, show his romantic side on the smash Mary J. Blige collab “All I Need” and just have fun with Redman on the duo’s How High films and Blackout! albums all without missing a beat. More than 20 years after making his debut on the Wu’s Enter the 36 Chambers, he’s not done: a standout verse on Conway the Machine’s 2020 single “Lemon” found him just as sharp as ever. “I’m not a player, I just crush a lot,” Big Pun declared on the 1998 smash hit “Still Not a Player” from his debut album Capital Punishment. A remix of the album’s lead single “I’m Not a Player,” it flipped a Brenda Russell sample into a salsa-adjacent jam and cemented the Big Punisher as a commercial force. After the album went platinum, he became the first Latino solo rap act to sell over a million copies. Capital Punishment also topped the Top R&B/Hip-Hop Albums chart and peaked at No. 5 on the Billboard 200, due to Big Pun’s expert wordplay, vivid storytelling and ability to craft catchy hooks. The late rapper continues to be an inspiration to waves of rappers, particularly Bronx-bred MCs like Fat Joe, Remy Ma and Cardi B. Q-Tip, co-founder of the alternative hip-hop group A Tribe Called Quest, has been on point for over 30 years. He’s the artistic, esoteric and philosophical rapper who juxtaposed the streetwise and humble delivery of the group’s late Phife Dawg. Tip’s laid-back, smooth-talkin’ flow came peppered with immortal lyrics in playful songs like “Bonita Applebum.” His body of work further established the creative link between hip-hop and jazz. He set a new benchmark for erasing musical boundaries and ignoring popular trends, influencing a host of generation-next talents like Pharrell Williams, Kanye West and Tyler, the Creator. Alongside Phife, Ali Shaheed Muhammad and Jarobi White, ATCQ — also part of the Native Tongues collective (Queen Latifah, De La Soul, Jungle Brothers, Monie Love) — claimed No. 1 twice on the Billboard 200, while all six of the group’s albums were certified either gold or platinum. Listening to Black Thought is like enrolling in a masterclass on lyricism. Tariq Trotter, co-founder of the legendary Roots crew and the long-running Philly band’s lead MC, still amazes rap fans with electrifying stream-of-consciousness radio freestyles and brilliant live performances. When Black Thought joins forces with supreme lyricists such as Eminem (“Yah Yah”), Royce da 5’9″ (“Rap on Steroids”), Big Pun (“Super Lyrical”) or Joey Bada$$ and Russ (“Because”), the combination elevates the craft and technique of rap to another level. Rapping since 1993, Black Thought continues to grow more polished as one of the genre’s greatest MCs. “You Got Me,” the Erykah Badu-featuring crossover duet from the Roots’ platinum-selling album Things Fall Apart, remains a shining moment thanks to his rich baritone and riveting storytelling. As a wellspring of talent, Virginia has had an outsized influence on popular music. And no conversation about that can be had without mentioning The Neptunes, Clipse and the latter duo’s enduring breakout star Pusha T. After reaching legendary status in the ‘00s alongside older brother Gene aka No Malice, Pusha later began grinding out a solo career, signing with G.O.O.D. Music in 2010 and kicking off the label relationship with a guest feature on “Runaway” from Kanye West’s My Beautiful Dark Twisted Fantasy that same year. From there, Pusha became a fashion icon and served as president of West’s label, while continuing to release well-received solo albums full of vivid descriptions of his drug hustling and emotional struggles. 2022’s It’s Almost Dry earned the rapper his first No. 1 on the Billboard 200. The timeless Ms. Lauryn Hill straddles the line between Motown soul, boom bap, R&B, jazz and rap — defying convention and introducing a new standard for genre-bending rappers. Whether it’s her work on Fugees’ blockbuster 1996 album The Score or her game-changing 1998 debut The Miseducation of Lauryn Hill, she has influenced a host of hip-hop artists over the decades. Her melodic rapping on “Doo Wop (That Thing)” led to the song becoming the first by a female rapper to top the Hot 100. Earning 10 Grammy nods for Miseducation, Hill won five, including best new artist and album of the year. Even if Hill never drops a proper sophomore album, she remains a peerless talent who can still outrap anyone — as proven by her memorable verse on Nas’ KD2 gem “Nobody” in 2021. Lil Kim’s raunchy, vicious rhymes turned hip-hop upside down. Under the tutelage of the late Notorious B.I.G., Queen Bee debuted as a member of Junior M.A.F.I.A., becoming an icon following her titillating 1996 solo debut album Hard Core. Certified double-platinum, the album spun off three consecutive No. 1s on Billboard’s Hot Rap Songs chart — “No Time,” “Not Tonight (Ladies Night Remix)” and “Crush on You” — making her the first female rapper to do so. Her unapologetic tone shattered barriers for women in hip-hop, allowing them to be more sexually expressive and liberated. While Kim’s movie idol looks and provocative fashion sense commanded eyes, her gruff delivery and punchy rhymes brought fear and respect from rival MCs, as evidenced by her searing features on Bad Boy classics like The LOX’s “Money, Power & Respect” and Diddy’s “It’s All About the Benjamins.” After a disappointing debut effort with 2001’s I’m Serious, T.I. regained his swagger with his 2003 breakthrough album Trap Muzik. Through vivid street tales and bombastic production, Tip became one of trap’s forefathers, alongside fellow heralded ATL stars such as Jeezy and Gucci Mane. Even when dueling alongside rap greats like Jay-Z (“Swagger Like Us”), Eminem (“Touchdown”) and Kanye West (“Welcome to the World”), T.I. expunged any doubt about his MC credibility via his grit and lyrical precision. One of hip-hop’s pioneering polymaths, T.I. has earned three Grammys and seven top five Billboard 200 albums and forged a successful film and TV career. Busta Rhymes’ scene-stealing feature on A Tribe Called Quest’s 1992 single “Scenario” introduced a Brooklyn MC that took over the late ‘90s and early ’00s with his animated voice and exuberant rhymes. The Coming, Rhymes’ 1996 major label debut album, couldn’t have sported a better title to describe his explosive entrance. His imaginative vision also vaulted him into hip-hop lore, as he and director Hype Williams whipped up some of the genre’s most innovative videos, including “Put Your Hands Where My Eyes Could See” and “Dangerous.” Aside from his puckish verses, Rhymes brought his A game when paired with R&B greats such as Mariah Carey (“I Know What You Want”) and Janet Jackson (“What’s It Gonna Be?!”). "
canon2 = "To describe Chuck D as a rapper would be as brazen an understatement as labeling Jimi Hendrix a guitarist. On Public Enemy’s 1987 “Rebel Without a Pause,” the seminal group’s mighty orator — backed by rap’s archetypal hype man in Flavor Flav — made it clear that hip-hop would never be the same. “Impeach the president, pulling out my RAY-GUN/ Zap the next one, I could be your shogun,” he exploded over the Bomb Squad’s revolutionary, deconstructed production. Chuck held everyone’s feet to the fire: crack dealers in the Black community (“Night of the Living Baseheads”); purveyors of systemic racism (“Fight the Power”); a ravenous press (“Welcome to the Terror”) and greedy corporations (“Shut ‘Em Down”). Embracing his nickname as rap’s “toxic king,” Future and his syrupy flows were a driving force in Atlanta’s claiming of the hip-hop throne in the 2010s. Pluto’s epic mixtape three-peat of Monster, Beast Mode, and 56 Nights followed by the critically acclaimed DS2 and his What a Time to Be Alive joint project with Drake — all within the calendar year of 2015 — could be the best use of 365 days that rap has ever seen. And that was only the tip of the iceberg: After releasing 2022’s I Never Liked You, featuring current Grammy nominee and No. 1 single “Wait for U” with Drake and Tems, he now boasts eight No. 1 albums and 10 top 10 Hot 100 hits to his name. Over a decade into his prolific career, Future Hendrix is more commercially relevant than ever: the ATL trap legend became the only artist to appear on the Hot 100 every week throughout 2022. Under his now-retired moniker Mos Def, Brooklyn native Bey broke into the underground scene on Rawkus Records’ influential 1997 compilation Soundbombing with a charismatic presence that recalled the early ‘80s chief rocker days of Busy Bee Starski. Yet on the 1998 collaborative milestone Mos Def & Talib Kweli Are Black Star, Bey proved to be so much more — a revelation further amplified on the unmoored race man’s gold-certified 1999 solo debut Black on Both Sides. He could effortlessly maneuver from bawdy around-the-way storytelling (“Ms. Fat Booty”) to the inner-spiritual fight for Black liberation (“UMI Says”). After a return-to-form with 2009’s The Ecstatic, the reinvigorated MC-turned-actor (Brown Sugar, Dexter) changed his name to Bey in 2011. Lonnie Rashid Lynn was seemingly carrying the hopes, dreams and aspirations of the entire Chicago rhyme community on his back with 1992’s Can I Borrow a Dollar? It took a minute, but Common (then Common Sense) finally garnered national praise after launching one of the most celebrated three-album runs in rap history: 1994’s Resurrection, 1997’s One Day It’ll All Make Sense and 2000’s Like Water for Chocolate. Few lyricists possess the dexterity to pull off a brilliant allegory about the history of hip-hop on “I Used to Love H.E.R.” while also firing off a savage takedown of a fearsome Ice Cube on “The B–ch In Yoo,” and then later becoming a Grammy-nominated hitmaker on “The Light.” And in 2017, Common became the first rapper to get the first 3/4 of the way to an EGOT by winning an Emmy (“Letter to the Free,” from 13th), Grammy (“Love of My Life” with Erykah Badu) and Oscar (“Glory” with John Legend, from Selma). Gucci Mane went from teenage drug dealer to Atlanta rap deity in the 200s, with sedated raps and a Hollywood story Spike Lee couldn’t script. Guwop initially emerged alongside one-time adversary Jeezy with their 2005 collaboration “So Icy,” before things went south between them. After serving a two-year prison sentence in 2016 for firearm possession by a convicted felon, a rejuvenated Guwop subsequently went on a mainstream run like he was a new artist. The trap pioneer’s workaholic mindset led to his flooding the streets with north of 70 mixtapes, and his hunger for greatness has yet to wane as he enters his 40s. His robust output yielded a record 20th top 10 entry on the Top Rap Albums chart in 2019, claiming the title belt from Tech N9ne. Prior to that in 2016, Mane scored his first Hot 100 No. 1 as a guest on Rae Sremmurd’s “Black Beatles.” Ludacris became Atlanta’s first crossover superstar of the 2000s, dominating the airwaves and the clubs with an arsenal of Dirty South anthems at his disposal, coupled with his uniquely charismatic twang. Luda’s witty rhymes and slick pop culture references felt as gargantuan as the famously oversized arms he rocked in 2004’s “Get Back” video. Five No. 1s on the Billboard Hot 100 and another four on the Billboard 200 albums chart are nothing to scoff at, as Ludacris’ chart resumé can “stand up” to just about any of his rap contemporaries. The Disturbing Tha Peace label co-founder’s industry acclaim finally caught up to his mainstream appeal when he took home the Grammys for best rap album and best rap song Grammys in 2007, thanks to the introspective Release Therapy and its Pharrell-featuring smash lead single “Money Maker.” Dr. Dre would be the first to tell you that he is not a traditional MC. During his 30+-year career, the global best-selling producer, often cited as hip-hop’s GOAT behind the boards, boasts a treasure trove of lyricists who have written for him, from Ice Cube, the D.O.C. and Snoop Dogg to Eminem, Jay-Z and Kendrick Lamar. However, Andre Young’s legacy as a lead rapper on three separate era-defining albums — 1988’s Straight Outta Compton with N.W.A. and his solo-headlined studio albums The Chronic and 2001 — can’t be denied. On Chronic, his 1992 triple-platinum-certified gangsta rap opus, Dre also exhibits an effortless delivery on the mic, with a commanding voice that’s one of the genre’s most recognizable. And hearing him master the double-time flow on 1999’s Slim Shady-featuring “Forgot About Dre,” from the six-times-platinum-certified masterpiece 2001, is still a shock to the system. San Francisco’s Bay Area has always been a key birthplace for unique hip-hop talent, from MC Hammer and Tupac Shakur to Too $hort. But few rappers, if any, have repped the Bay longer or stronger than E-40. With one of the deepest catalogs in hip-hop history — including 18 top 10s on Billboard’s Top Rap Albums chart — Forty Water personifies the hustle of forging a successful independent career. His flamboyant rhymes helped him become one of the first West Coast rappers to sign a major deal when he signed with Jive Records in the early ‘90s. And he hasn’t gone more than four years without releasing an album since 1993. E-40’s contributions, including his 2006 top 10 rap hit and hyphy movement anthem “Tell Me When to Go,” helped bring the Bay Area into the hip-hop’s mainstream. Redman’s rugged yet humorous rhymes and infectious personality make him one of hip-hop’s most beloved MCs. In fact, Reggie Noble led off Eminem’s list of the greatest MCs within the latter’s pump-up perennial “Till I Collapse.” With EPMD’s Erick Sermon as a mentor, Redman (aka Funk Doc) exceeded expectations in 1992 when his debut studio project Whut? Thee Album became an instant hip-hop classic with its early use of funk samples and hilarious punchlines. Redman’s popularity skyrocketed when he joined forces with Method Man in the late ‘90s and released the first of their two albums together — Blackout!, which debuted at No. 3 on the Billboard 200. Fans couldn’t get enough of the weed-smoking, carefree duo’s debauchery, which culminated in the 2001 cult classic feature film, How High, starring the pair and named after one of their enduring ‘90s hits. As the legendary rap duo UGK, Bun B and the late Pimp C kept the spotlight on Texas while taking the baton from the Scarface-led triad Geto Boys. However, the Port Arthur native born Bernard James Freeman twangy flow ended up being far more influential than his initial underground career aspirations. A random call from Jay-Z in 1999 ended up shattering regional exposure barriers when Jay’s braggadocious “Big Pimpin’” earned UGK its first top 40 Hot 100 hit. After Bun proved his chops as a solo artist with Pimp C behind bars, UGK reunited to craft its magnum opus with 2007’s Billboard 200-topping Underground Kingz, released just months before Pimp C’s death at 33 in 2007. The set was powered by the majestic “Int’l Players Anthem” featuring OutKast. Now a rap elder statesman, Bun keeps a keen eye on next-gen talent, offering key co-signs to such contemporary stars as Drake and Kodak Black. “Who you calling a b–ch?!”  And with those iconic words from the powerful 1993 anti-domestic violence statement “U.N.I.T.Y.,” Queen Latifah cut through all the misogynistic noise. This was nothing new for New Jersey’s own Dana Owens: With her essential 1989 debut album All Hail the Queen, she not only delivered the classic Monie Love-featured Black feminist anthem “Ladies First,” she also got the party started (“Come Into My House”) and took out wack MCs (“Wrath of My Madness’). Latifah’s subsequent jump into Hollywood proved to be just as impressive, earning accolades for her television work (Living Single) and an Oscar nod (the 2002 musical Chicago), and making her one of the first MCs with the star power to prove how deeply a rapper could become embedded in American pop culture, even outside of the world of hip-hop. Ice-T’s harrowing “6 N the Mornin’” (1986) forever stamped the West Coast on the hip-hop map. To the Moral Majority, the Godfather of gangsta rap — who set the table for West Coast MCs ranging from N.W.A and Snoop Dogg to The Game and the late Nipsey Hussle — was a walking Parental Advisory sticker. Yet like “Colors,” Ice’s masterful 1988 first-person account of the Crips and Bloods gang violence that gripped Los Angeles, street manifestos like Rhyme Pays (1987), Power (1988) and The Iceberg /Freedom of Speech… Just Watch What You Say! (1989) were eloquent hood testimonies. When his metal band Body Count’s incendiary “Cop Killer” (1992) nearly derailed his career, Ice also helped open a new lane for hip-hop: acting (New Jack City, Law & Order: SVU). Jadakiss’ signature laugh and “ah-ha” squeal have signaled for decades that a lyrical masterclass was on the way. The LOX frontman fostered his hip-hop breakout with a co-sign from The Notorious B.I.G., then stomped his way to prominence with his Timberlands and hard-nosed raps. His lyrical intensity and fearless aggression are battle-tested, as he sparred with the likes of 50 Cent and Beanie Sigel in the 2000s before single-handedly dismantling Dipset in a 2021 Verzuz battle. Jada checks off just about every box as a member of hip-hop’s hall of fame — though a bona fide classic solo album from the Yonkers legend could have vaulted him inside this list’s top 20. In 1987, MC Lyte bumrushed her way into hip-hop’s boys club with a throat-grabbing voice and dynamic lyricism that seemed well beyond her 16 years. The Brooklyn teen’s opening salvo was “I Cram to Understand U (Sam),” a song that spoke about the perils of falling in love with a crack addict. Even the title of the original queen of rap’s 1988 seminal debut album, Lyte as a Rock, was metaphorically heavy. From battle rhyming (“Shut the Eff Up-Hoe”) to heartfelt storytelling (“Poor Georgie”), Lyte roared into the ‘90s racking up three gold singles — most notably the Puffy Combs-produced “Cold Rock a Party” (1996), featuring an then-up-and-coming Missy Elliott. Before Melle Mel’s game-changing run with Grandmaster Flash & The Furious Five, rap was still constrained by its “yes, yes y’all!” park jam origins. Then came “The Message” (1982), hip-hop’s seismic GOAT recording, elevated by the rapper born Melvin Glover and his vivid ghetto portrait. A year later, the first universally hailed God MC demystified the allure of the Big Apple (“New York, New York”) and distilled the crisis of the cocaine epidemic (“White Lines [Don’t Don’t Do It]”) before crashing pop radio with his 1984 appearance on Chaka Khan’s Grammy-winning, No. 3-peaking Hot 100 hit “I Feel for You” — a pivotal early crossover moment between the hip-hop and R&B worlds. “I’m drivin’ Caddy, you fixin’ a Ford,” Joseph Simmons boasted on Run-DMC’s 1984 breakthrough “Rock Box.” While DMC was the majestic voice and the late DJ Jam Master Jay the heartbeat of the epochal ’80s rap trio, Run was the undisputed star. He led Run-DMC to historic heights, as they became the first hip-hop group to flex B-Boy minimalism to the masses (“Sucker MC’s”), go gold (1984’s King of Rock), appear on MTV, reach multiplatinum status (with 1986’s landmark Raising Hell, punctuated by the boundary-breaking Aerosmith collab “Walk This Way”), headline arena tours and ink a major endorsement deal (Adidas). After finding God, Rev. Run reinvented himself as a reality star in the 2005 MTV series Run’s House. Rozay’s “Hustlin’” dreams turned to gold when he rattled the cages of rap’s mainstream with his 2006 summer anthem and pledged his allegiance to Def Jam and its ex-president Jay-Z. Ross demanded attention through his deep gruff-and-grunting ad-libs, colorful street tales and inside looks at Miami’s lavish lifestyle, plus his incredible ear for production — more appetizing than a lemon-pepper Wingstop order. His catalog reigns supreme, especially in the late 2000s and 2010s, as he doled out gems such as Trilla, Deeper Than Rap and God Forgives, I Don’t. Ross has notched 58 Billboard Hot 100 entries spanning three decades, but his legacy doesn’t end there: His Maybach Music Group laid the tarmac for Meek Mill and Wale to take off and become titans in the 2010s. "

#combining text blocks
canon_all = canon_top10 + canon1 + canon2

#tokenizing
canon_token = nltk.word_tokenize(canon_all)

#now time to get rid of our stop words
stop_words = nltk.corpus.stopwords.words('english')
filtered_list = []
for word in canon_token:
    if word.casefold() not in stop_words:
         filtered_list.append(word)
frequency_distribution = FreqDist(filtered_list)
print(frequency_distribution.most_common(20))

#good start, but lots to add to our stop words here
new_stop_words1 = [',', '.', '’', '“', '”', '(', ')', 'rap', 'hip-hop', 'album', '—', ':', 'albums', 'rapper', 'one', 'like', 'also']
stop_words.extend(new_stop_words1)
filtered_list = []
for word in canon_token:
    if word.casefold() not in stop_words:
         filtered_list.append(word)
frequency_distribution = FreqDist(filtered_list)
print(frequency_distribution.most_common(50))

#much better, but let's refine this further
new_stop_words2 = ['&', 'Hot', 'career', '1', '‘', '100', 'became', '–', '200', 'year', 'rhymes', 'MC', 'late', 'early', 'three', 'top', 'rappers', 'MCs', 'still', 'could', 'music', 'label', 'single', 'best', 'artist', 'including', '?', 'even', 'R']
stop_words.extend(new_stop_words2)
filtered_list = []
for word in canon_token:
    if word.casefold() not in stop_words:
         filtered_list.append(word)
frequency_distribution = FreqDist(filtered_list)

#a few stubborn characters aren't filtering out, so did 23 instead of 20 to capture the true top 20
print(frequency_distribution.most_common(23))

#now let's contrast the whole list with the top 10 of the canon
canon_top10_token = nltk.word_tokenize(canon_top10)

#now time to get rid of our stop words
filtered_list = []
for word in canon_top10_token:
    if word.casefold() not in stop_words:
         filtered_list.append(word)
frequency_distribution = FreqDist(filtered_list)
print(frequency_distribution.most_common(20))

#much of our work is done, but let's kick out a few bad words
new_stop_words3 = ['time', 'releasing', 'studio', 'later', 'R', 'success', 'way', 'years']
stop_words.extend(new_stop_words3)
filtered_list = []
for word in canon_top10_token:
    if word.casefold() not in stop_words:
         filtered_list.append(word)
frequency_distribution = FreqDist(filtered_list)

#one stubborn character remains, so did 21 instead of 20 (similar to before)
print(frequency_distribution.most_common(21))

#lemmatizing the entire selection so I can look for collocations
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in canon_token]
new_text = nltk.Text(lemmatized_words)
print(new_text.collocations(50))

#collocations are interesting, but most proper nouns. so, they offer limited utility. 
#there is ambiguity with some of these remaining results that will be insightful no matter what, but warrant investigation. 
#could use chunking, chinking or dispersion plots for this, but a concordance will be more informative
text50 = Text(canon_token)
print(text50.concordance('first'))

#very informative (notes in portfolio). let's do a few more of these on some of the more ambiguous words
print(text50.concordance('Black'))
print(text50.concordance('Records'))
print(text50.concordance('genre'))
print(text50.concordance('Big'))