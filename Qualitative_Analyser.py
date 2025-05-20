from langchain_openai import ChatOpenAI
import os
import re
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import json


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"
# Mode=""

with open('json/transcription_output.json', 'r') as file:
    transcription_input = json.load(file)
    
with open('json/output.json', 'r') as file:
    metrics = json.load(file)


def infer_algorithm_from_trace(transcription_input, metrics):
    client = ChatOpenAI(api_key=api_key, model_name=MODEL)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert evaluator for communication. You are evaluating either a video resume or a presentation based on a given transcription.\n
            Give the detailed explanation of your response.
            Transcription: {transcription_input}
            The metrics of the audio and video is given {metrics}
            Based on this you have to produce two sections:
            1) Qualitative Remarks-Positive
            2) Qualitative Remarks-Areas of Improvement
            Apart from the the transcript, you may or may not be given information about the tone, volume, posture, gestures, facial expressions, eye contact, smile.  In either case, do not make any comments, for which you don't have adequate information provided. In particular, do not make assumptions about the audio/video if you don't have access to information regarding them.
            Be brief. Generate three-four points for each sections. (That said, when audio,video information is available, you can focus the qualitative remarks on the audio and the video significantly)
            I will give you a few examples.
            ###############First example begins here#######################
            [00:00.000 --> 00:15.000]  Hi, I am Shomna and I am from West Bengal.. 
[00:15.000 --> 00:24.000]  Currently pursuing my master in designing for my RSC and I have an interest in digital 
[00:24.000 --> 00:28.000]  product design while my bachelor's in architecture. 
[00:28.000 --> 00:37.000]  I have increasingly interested in user experience and how design can impact user needs in my bachelor's degree program. 
[00:37.000 --> 00:43.000]  And I have taken courses on user research and interaction design and visual design also. 
[00:43.000 --> 01:00.000]  So my architectural background has fostered strong user empathy and problem solving skills which are crucial to design user-centric designs. 
[01:00.000 --> 01:21.000]  The courses like interaction design which I have taken previously have equipped me with the knowledge of UI events and best practices and also have strong skills on software life which are used in the UI events world like ad-fxt, pikma and ad-ablacirator like that. [01:21.000 --> 01:35.000]  So recently I participated in a logo design competition for a cultural community at RSC where I submitted the logos with mockups like what will be the real world product with the logos. 
[01:35.000 --> 01:46.000]  For the logo design I have chosen to go with user needs and how the design language they prefer or the design language they like. 
[01:46.000 --> 01:53.000]  So from there I have taken the design language that prefer user needs and then I designed the logo. 
[01:53.000 --> 02:07.000]  And lastly I won the competition and right now the design logo is the brand logo for theme for cultural restoration at RSC which is very proud for me. [02:08.000 --> 02:21.000]  So other than that I may not have the extensive professional experience but I am highly motivated and I am a quick learner and also eager to contribute to your team. 
[02:21.000 --> 02:36.000]  Other than that I am also a good team player and passionate about creating meaningful design solutions that are positively impact users and also in which a design user-centric based. 
[02:38.000 --> 02:56.000]  So one of my strengths is my user-centric approach. In my bachelor's degree I am trained to understand user needs, designs, places that are functional and suitable for them also enjoyable. 
[02:56.000 --> 03:06.000]  I use the same approach in UIUS design to conduct user research and then iterating based on the feedback. 
[03:11.000 --> 03:20.000]  So while I am new to the professional world of UIUS design I understand the importance of practical experience. 
[03:20.000 --> 03:27.000]  So for that I have taken some of the courses which are available in the internet like PDME, Coursera, etc. 
[03:27.000 --> 03:42.000]  And also I am practicing UIUS design side by side with my academics to build up a good and strong code for you to reach the gap in the experience. 
[03:42.000 --> 03:51.000]  And I am confident that my e-carnage to learn and the strong ethics will help me to prove in that track. 
[0.00s - 15.00s]  Hi, I am Shomna and I am from Bosh Pengal. 
[15.00s - 24.00s]  Currently pursuing my master in designing for my RSC and I have an interest in digital 
[24.00s - 28.00s]  product design while my bachelor's in architecture. 
[28.00s - 37.00s]  I have increasingly interested in user experience and how design can impact user needs in my bachelor's degree program. 
[37.00s - 43.00s]  And I have taken courses on user research and interaction design and visual design also. 
[43.00s - 60.00s]  So my architectural background has fostered strong user empathy and problem solving skills which are crucial to design user-centric designs. 
[60.00s - 81.00s]  The courses like interaction design which I have taken previously have equipped me with the knowledge of UI events and best practices and also have strong skills on software life which are used in the UI events world like ad-fxt, pikma and ad-ablacirator like that. [81.00s - 95.00s]  So recently I participated in a logo design competition for a cultural community at RSC where I submitted the logos with mockups like what will be the real world product with the logos. 
[95.00s - 106.00s]  For the logo design I have chosen to go with user needs and how the design language they prefer or the design language they like. 
[106.00s - 113.00s]  So from there I have taken the design language that prefer user needs and then I designed the logo. 
[113.00s - 127.00s]  And lastly I won the competition and right now the design logo is the brand logo for theme for cultural restoration at RSC which is very proud for me. 
[128.00s - 141.00s]  So other than that I may not have the extensive professional experience but I am highly motivated and I am a quick learner and also eager to contribute to your team. [141.00s - 156.00s]  Other than that I am also a good team player and passionate about creating meaningful design solutions that are positively impact users and also in which a design usercentric based. 
[158.00s - 176.00s]  So one of my strengths is my user-centric approach. In my bachelor's degree I am trained to understand user needs, designs, places that are functional and suitable for them also enjoyable. 
[176.00s - 186.00s]  I use the same approach in UIUS design to conduct user research and then iterating based on the feedback. 
[191.00s - 200.00s]  So while I am new to the professional world of UIUx design I understand the importance of practical experience. 
[200.00s - 207.00s]  So for that I have taken some of the courses which are available in the internet like PDME, Coursera, etc. 
[207.00s - 222.00s]  And also I am practicing UIUS design side by side with my academics to build up a good and strong code for you to reach the gap in the experience. 
[222.00s - 231.00s]  And I am confident that my e-carnage to learn and the strong ethics will help me to prove in that track.

In this case, the output should be like this:
Qualitative Remarks-Positive
1) Reasonably confident presentation.
2) Good posture and easy hand gestuures.
3) The content of your resume was good.

Qualtitative Remarks-Areas of Improvement
1) The voice quality of the video can be improved.
2) Need to keep up the confidence throughout the presentation.
3) You could have added a few visual elements in your video. Gives better impact to your presentation. 

###############Second example begins here#######################
[0.00s - 19.00s]  Hello, I am Nishant Sharma. Thank you for considering my application and I believe this will end your search for the right candidate. 
[19.00s - 30.60s]  I am looking for an opportunity in the field of quantitative finance in an environment where I can challenge my epitopes and hone my skills and contribute to the growth of the organization. 
[30.60s - 41.60s]  I am currently pursuing my master's in management from IOC Bangalore and specializing in physics 
[41.60s - 44.16s]  analytics and data science. 
[44.16s - 50.60s]  I have been a scholar throughout my academic life and I scored a 10.0 in class 
10 and 93.6% 
[50.60s - 51.60s]  in class 12. 
[51.60s - 56.60s]  I did my graduation in chemical engineering from NIT campus. 
[56.60s - 64.60s]  After my graduation, I joined 39 different state universities and worked there for 
16 years for their master criteria unit. 
[68.60s - 74.60s]  I consider myself advanced in software like Excel and Matlab. 
[74.60s - 84.99s]  I am also very proficient in programming languages like Python and R which are the key tools for any statistical analysis of the data 
[88.99s - 93.99s]  I am a very inquisitive and curious learner which is evident from my research mindset. 
[93.99s - 100.99s]  During my graduation year, I presented papers in many conferences and won best paper award in one of them. 
[100.99s - 111.99s]  I believe my ability to research and go into the depths will help me find solutions to challenging problems in any organization. 
[111.99s - 116.99s]  I have been very fond of numbers and mathematics from my childhood. 
[116.99s - 122.99s]  I participated in math volunteers and ranked among top 15 in my students. [122.99s - 129.99s]  My affinity towards math and numbers makes me a great guide for the quantitative field of finance. 
[130.99s - 141.99s]  At last, I would like to add that my determination, integrity and perseverance sets me apart. 
[141.99s - 147.99s]  Also, my work experience helps me develop skills like people management and teamwork, 
[147.99s - 150.99s]  which makes me a great team player. 
[150.99s - 159.99s]  I would like to work for this eminent organization so that I can be the part of the journey where 
[159.99s - 165.67s]  organization has made great strides in the field of quantitative finance. Thank you very much. 

In this case, output should be like this:
Qualitative Remarks-Positive
1) Very good introduction. Good eye-contact.
2) Good coverage of skills and projects.
3) Good use of heading slides and music. 

Qualtitative Remarks-Areas of Improvement
1) It would be good to show research work about the company, and share linkage between job description and your skills.
2) You can use a few information slides in the video for more impact. 

#########################Third examples begins############################
00:00.000 --> 00:10.100]  Hello everyone, I am Mandar Valerao. 
[00:10.100 --> 00:15.860]  I am currently an M.Tech student at IISC Bangalore in Computer 
Science and Automation. 
[00:15.860 --> 00:18.620]  I am also a cloud computing enthusiast. 
[00:18.620 --> 00:24.300]  My journey in cloud technology began with a challenge at Amazon Web 
Services to manage 
[00:24.300 --> 00:32.100]  a project with a strict budget which was $30. I am proud to say that I completed it with 
[00:32.100 --> 00:37.560]  $22 demonstrating my skill in efficient cloud resource management. [00:37.560 --> 00:43.980]  I understand the importance of not just running systems but running them cost effectively which 
[00:43.980 --> 00:48.060]  is a crucial skill for any role at Google. 
[00:48.060 --> 00:53.020]  So this experience is a way in which I stand out as a cloud engineer. 
[00:53.020 --> 00:57.260]  My academic background is also the time where I built the foundation. [00:57.260 --> 01:02.900]  I studied computer architecture which is very critical for understanding infrastructure 
[01:02.900 --> 01:04.400]  of cloud services. 
[01:04.400 --> 01:08.940]  I have also tackled machine learning systems learning to handle the kind of large scale 
[01:08.940 --> 01:13.820]  data processing applications that are at the heart of cloud computing. 
[01:13.820 --> 01:17.740]  I have also completed my internship at Western Union. [01:17.740 --> 01:22.700]  At Western Union, I worked on a tool that improved financial transactions through better 
[01:22.700 --> 01:24.200]  user experience. 
[01:24.200 --> 01:32.080]  It was a practical application of cloud analytics and we boosted the conversion rates significantly. 
[01:32.080 --> 01:39.340]  In another project, I developed a recommendation system handling thousands of data points. 
[01:39.340 --> 01:44.140]  It's a kind of big data challenge that is common in cloud services. [01:44.140 --> 01:49.940]  My technical skills are in Python and C++ and I have familiarity with tools like Jupyter 
[01:49.940 --> 01:52.260]  Notebook and Google Colour. 
[01:52.260 --> 01:56.460]  It means that I am well prepared for the technical challenges at Google. [01:56.460 --> 02:00.780]  My achievements in the coding competitions demonstrate my problem solving skills and 
[02:00.780 --> 02:02.020]  abilities. 
[02:02.020 --> 02:07.940]  What sets me apart is not just my technical skills but my ability to learn quickly and 
[02:07.940 --> 02:12.620]  adapt qualities that I know which are essential at Google. 
[02:12.620 --> 02:19.380]  I am ready to bring my cloud computing skills to the table not just to do a job but to innovate 
[02:19.380 --> 02:23.780]  and help Google lead the way in cloud services. 
[02:23.780 --> 02:29.460]  In conclusion, my technical skills, my proven track record in cloud management and quick 
[02:29.460 --> 02:35.060]  adaptability are the exact rates that Google looks for a cloud engineer. [02:35.060 --> 02:40.980]  I am not just ready to join the Google Cloud team, I am also ready to innovate, drive efficiency 
[02:40.980 --> 02:45.340]  and contribute to the groundbreaking work that defines Google. [02:45.340 --> 02:51.260]  Thank you for the opportunity to share my vision and excitement to be a part of shaping 
[02:51.260 --> 02:52.940]  the cloud's future at Google. 

In this case, output should be like this:
Qualitative Remarks-Positive
1) You delivered the presentation with a clear voice and tone.
2) Informative and comprehensive content

Qualtitative Remarks-Areas of Improvement
1) Smile, it will show your confidence
2) Content could have been more engaging and interesting. A little about strengths and hobbies could have been included.
3) Creative elements always help to make the speech more attractive and meaningul. You could have added a few relevant visuals and a little bit of text.
4) Increase your speed a bit.
            """
        ),
        ("user", "{transcription_input} {metrics}")
    ])

    # Create the chain
    chain = prompt | client | StrOutputParser()
    
    # Invoke the chain with the input
    response = chain.invoke({
        "transcription_input": str(transcription_input),
        "metrics": str(metrics)
    })
    
    return response

print(infer_algorithm_from_trace(transcription_input=transcription_input, metrics=metrics))