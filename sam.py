import pyttsx3 
import subprocess 
import webbrowser 
import smtplib 
import json
import random
import operator
import speech_recognition as sr
import wikipedia
import datetime
import wolframalpha
import os
import sys
import time
import requests
import shutil
from playsound import playsound #pip install playsound
import platform #an-inbuilt module
from urllib.request import urlopen
from clint.textui import progress  #pip install clint
from ecapture import ecapture as ec  #pip install ecapture
from bs4 import BeautifulSoup #pip install urllib
import cv2  
import openai
import  face_recognition as fr
from apikey import api_data
import numpy as np

openai.api_key=api_data


completion=openai.Completion()

def Reply(question):
    prompt=f'Jeevan : {question}\n Sam: '
    response=completion.create(prompt=prompt, engine="text-davinci-002", stop=['\Jeevan'], max_tokens=200)
    answer=response.choices[0].text.strip()
    return answer

def Command():

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold =  0.6
        audio = r.listen(source)
    try:
        cmd = r.recognize_google(audio, language="en-IN")
        print('User: ' + cmd + '\n')

    except sr.UnknownValueError:
        cmd = Command()
    
    return cmd
'''
def Command():
    return input(str('User: ')).strip()
'''



name = 'SAM'
engine = pyttsx3.init()
idle = False

client = wolframalpha.Client('9L5757-TQKUTWJGA2')

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 135)


#face rcognition part 
faces = [
    cv2.imread("principal.jpg"),
    #cv2.imread("known_face_2.jpg"),
    # Add more known faces if needed
]
index = 0  






def speak(audio):
    print(name +':' + audio)
    engine.say(audio)
    engine.runAndWait()

def Welcome2(name = "sir"):
    currentH = int(datetime.datetime.now().hour)
    if currentH >= 0 and currentH < 12:
        speak('Good Morning')


    if currentH >= 12 and currentH < 4:
        speak('Good Afternoon')


    if currentH >= 18 and currentH !=0:
        speak('Good Evening')


def Welcome():
    currentH = int(datetime.datetime.now().hour)
    if currentH >= 0 and currentH < 12:
        speak('Good Morning')

    if currentH >= 12 and currentH < 18:
        speak('Good Afternoon')

    if currentH >= 18 and currentH !=0:
        speak('Good Evening')
        


"""def Princi():
    currentH = int(datetime.datetime.now().hour)
    if currentH >= 0 and currentH < 12:
        speak('Good Morning principal mam')

    if currentH >= 12 and currentH < 4:
        speak('Good Afternoon principal mam')

    if currentH >= 18 and currentH !=0:
        speak('Good Evening principal mam')"""

def IdleMode(path = "principal.jpg", name = "Principal"):


    video_capture = cv2.VideoCapture(1)


    Jvn_image = fr.load_image_file(path) # Rename Your Detected Face Photo as Me.jpg And Keep Both Python File And This File In The Same Script
    Jvn_face_encoding = fr.face_encodings(Jvn_image)[0]

    known_face_encondings = [Jvn_face_encoding]
    known_face_names = [name] # Change It To Your Name

    def frcn(is_detected):
        if is_detected:
            speak(f"{name}'s Face Was Detected")


            webbrowser.open("http://192.168.137.115/bt")


            time.sleep(5)

            webbrowser.open("http://192.168.137.230/inline")


            time.sleep(5)
            speak(f"Hello {name}")
            speak("how do you want me to help you today?")
            return True

        else:
            #speak("Jeevan's Face Not Detected")
            pass

    
    while True:
        ret, frame = video_capture.read()
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        rgb_frame = frame[:, :, ::-1]

        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        face_detected = False

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = fr.compare_faces(known_face_encondings, face_encoding)

            name = {name}

            face_distances = fr.face_distance(known_face_encondings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_detected = True

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if frcn(face_detected):
            return True

        cv2.imshow('Face_Recognition', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


def IdleMode2(path = "jeevan.jpg", name = "Jeevan Suresh"):
    video_capture = cv2.VideoCapture(0)


    

    Jvn_image = fr.load_image_file("yashwanth.jpg") # Rename Your Detected Face Photo as Me.jpg And Keep Both Python File And This File In The Same Script
    Jvn_face_encoding = fr.face_encodings(Jvn_image)[0]

    known_face_encondings = [Jvn_face_encoding]
    known_face_names = [name] # Change It To Your Name

    def frcn(is_detected):
        if is_detected:
            speak(f"{name}'s Face Was Recognized")


            webbrowser.open("http://192.168.137.115/bt")


            time.sleep(5)

            webbrowser.open("http://192.168.137.230/inline")


            time.sleep(5)
            speak(f"Hello {name}")
            speak("Nice meeting you")
            speak("how do you want me to help you today?")  
            return True

        else:
            #speak("Jeevan's Face Not Detected")
            pass

    
    while True:
        ret, frame = video_capture.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        rgb_frame = frame[:, :, ::-1]

        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        face_detected = False

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = fr.compare_faces(known_face_encondings, face_encoding)

            name = "Unable To Recognize"

            face_distances = fr.face_distance(known_face_encondings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_detected = True

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if frcn(face_detected):
            return True

        cv2.imshow('Face_Recognition', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


    #result = ccv(faces, index)
"""    if result == True:
        Welcome(name = "princip")
        speak("pleasure to meet you mam")
        
"""        
        

spk = ['how can i help you today ?',"what can i do for you"]
speak(random.choice(spk))


var = False


if __name__ == '__main__':

    while True:
        '''  
        a = IdleMode()
        if a:
             Welcome2(name = "H O D mam ")
        '''





        if var:
            cmd = Command()
            cmd = cmd.lower()
        else:
            cmd = input("user:").lower()
            var = True

        
            
        if 'idle' in cmd or 'idle mode' in cmd or 'enter idle mode' in cmd or 'go idle' in cmd or 'go to idle mode' in cmd or 'go to idle' in cmd :
            speak('okay,entering idle mode')
            flag = IdleMode(path = "principal.jpg", name = "pricipal mam")
            #flag = IdleMode(path = "vp.jpg", name = "vice pricipal mam")
            #flag = IdleMode(path = "sathyaraj.jpg", name = "sathyaraj sir")
            #flag = IdleMode(path = "james.jpg", name = "james sir")
            #flag = IdleMode(path = "hm.jpg", name = "head misteress mam")
            #flag = IdleMode(path = "prathapharidas.jpg", name = "Prathap Haridoss Sir")
            #flag = IdleMode(path = "sankarasubramanian.jpg", name = "SankaraSubramanian Sir")
            #flag = IdleMode(path = "balasubramaniam.jpg", name = "Balasubramaniam Sir")
            #flag = IdleMode(path = "cschod.jpg", name = "h o d mam")
            #flag = IdleMode(path = "joanol.jpg", name = "Primary Ma'am")
            

            if flag:
                pass

        elif "standby" in cmd or "standby mode" in cmd:
            speak("okay")
            #flag = IdleMode2(name = "keshav ", path = "kesh.jpg")
            flag = IdleMode2(name = "jeevan", path = "jeevan.jpg")
            #flag = IdleMode2(name = "YAshwanth", path = "yashwanthinAuto.jpg")
            if flag:
                
               pass
                
                #Welcome2(name= "Jeevan")

        elif 'i am fine how are you' in cmd or 'i am fine' in cmd or 'i am good' in cmd:
            speak("good , im too fine")
        elif 'sam laugh' in cmd or 'sam laugh once more' in cmd or 'sam lol' in cmd or 'lol' in cmd or 'giggle' in cmd or 'laugh' in cmd or 'lol' in cmd  or 'laugh again' in cmd or 'laugh once again' in cmd or 'can you laugh again' in cmd or 'laugh once more' in cmd or 'entertain me' in cmd :
            speak("hahahahahaha")
                
            
        elif 'search' in cmd or 'google' in cmd or 'browse' in cmd or 'sam search' in cmd:
                speak("sorry  ,what should i search for")
                search= Command()
                speak("ok searching for "+search)
                webbrowser.open("https://www.google.com/search?client=firefox-b-d&q= "+search)
        elif 'open google'in cmd or 'google' in cmd or 'sam open google' in cmd:
            speak('okay,opening google')
            webbrowser.open('www.google.co.in')

        elif 'open gmail' in cmd or 'gmail' in cmd or 'sam open gmail' in cmd:
            speak('okay,opening gmail')
            webbrowser.open('www.gmail.com')

        elif "whats up" in cmd or 'how are you' in cmd or 'hi sam , how are you' in cmd:
            stMsgs = ['i am good as always','i am fine , thanks for asking' ,'it depends on your internet connection']
            speak(random.choice(stMsgs))

        elif 'email' in cmd or ' send email' in cmd or "mail" in cmd:
            speak('Who is the recipient? ')
            recipient = Command()

            if 'me' in recipient:
                try:
                    speak('What should I say? ')
                    content = Command()

                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.ehlo()
                    server.starttls()
                    server.login("Your_Username", 'Your_Password')
                    server.sendmail('Your_Username', "Recipient_Username", content)
                    server.close()
                    speak('Email sent!')

                except:
                    speak('Sorry ! I am unable to send your message at this moment!')

                    
        elif "stop moving" in cmd:
            speak("command recognized , stopping")
            webbrowser.open("192.168.137.115/stp")

        elif 'nothing' in cmd or 'die' in cmd or 'abort' in cmd or 'stop' in cmd or 'sam stop' in cmd or 'close' in cmd or 'sam close' in cmd or 'quit' in cmd:
            speak('okay')
            speak('Bye, have a good day.')
            sys.exit()
        
        elif "lgbtq" in cmd:
            speak("I  dontsupport LGBTQ")
            speak("even though I identify as a Robot")

        
        elif 'bye' in cmd or 'good bye' in cmd or 'tata' in cmd or 'buh bye' in cmd or 'bye sam' in cmd:
            speak('Bye, have a good day.')
            sys.exit()

        elif 'hey sam how are you' in cmd   or 'sam how was your day' in cmd or 'how was your day' in cmd or 'how is your day going' in cmd or 'how are you' in cmd:
            opt=[' i\'m good ','fine','good','i had a fine day']
            speak(random.choice(opt))

        elif "who created you" in cmd or "maker" in cmd or "who is your creator" in cmd or "who was your creator" in cmd or "who is the owner of sam" in cmd or 'by whom were you created' in cmd or 'who owns you' in cmd :
            speak('I am created by Jeevan Suresh and his team')
                
        elif "i want to buy something" in cmd  or "i want to buy things in online"  in cmd or "i want to buy things online" in cmd or "i want to buy a" in cmd or "order in online" in cmd:
            # can add somemore
            speak('Do you want me to open in amazon,flipkart or snapdeal')
            choice = Command()
            if choice.lower() == 'amazon':
                webbrowser.open_new_tab('http:\\www.amazon.com')
                speak('a new tab of amazon have been opened')
            elif choice.lower()=='flipkart':
                webbrowser.open_new_tab('http:\\www.flipkart.com')
                speak('a new tab of flipkart have been opened')
            elif choice.lower()=='snapdeal':
                webbrowser.open_new_tab('http:\\www.snapdeal.com')

            else:
                speak('sorry i am not familiar with it')

        elif "what is your name" in cmd or "can i know your name please " in cmd or "Hey sam what is your name" in cmd or 'hi  what is your name' in cmd or ' can i know your name' in cmd or ' hey what is your name' in cmd or "introduce yourself" in cmd:
                speak('MY name is sam')
                speak("I am a humanoid Robot")
                
        elif"What is your age " in cmd or "can i know your age please " in cmd or "how old are you" in cmd or "Hey sam what is your age" in cmd or 'age' in cmd or 'can i know your age' in cmd or ' hi can i know your age' in cmd or ' hey what is your age' in cmd:
                speak(" i was just born when Jeevan Suresh was pursuing his 9th grade")
                
        elif "where is your home" in cmd or "where were you born " in cmd or "where do you live " in cmd or 'where is your house' in cmd or  'in which place do you live' in cmd or 'where is your house located in ' in cmd or ' where is your home located in' in cmd or 'what is your address' in cmd or ' what is your house address' in cmd or 'what is your home addess' in cmd:
                speak("i Live in your internet connection")
                
        elif 'what is your job' in cmd or 'what is the job you are doing' in cmd or 'can i what work are you doing' in cmd or 'can you tell me what\'s your work' in cmd or 'are you carrying on any job' in cmd or "what can you do" in cmd:
                sts=['I Run all the artificial intelligence voice assistance service for the robot and I can perform general voice assistant tasks and provide results by browsing the web. I can also perform other computational tasks ']
                speak(random.choice(sts))
        elif 'who are you'in cmd or 'introduce yourself'in cmd:
            speak("I am sam , the humanoid robot and the voice Assistant")
            
        elif 'hey ,whats up' in cmd or 'how are you' in cmd:
            speak("i am good as always")
            
        elif 'turn on game mode'in cmd or 'play a game'in cmd or 'game'in cmd or 'i want to play a game' in cmd or 'open22222 a game' in cmd:
            speak("ok turning on game mode")
            speak("what game do you want to play")
            speak("choose either flappy bird , Ping pong")
            game = Command()
            game = game.lower()
            if (game == 'flappy bird' or game == 'flappybird'):
                    speak("ok , opening flappy bird")
                    os.system('main.py')
                    
            elif(game == 'pingpong' or game == 'ping pong'):
                    speak("ok opening pingpong")
                

                    webbrowser.open("https://editor.p5js.org/jeevansuresh2508/full/eqSkeFb71")
                    
                    
            else :
                    speak("that's an invalid choice")
                    
        elif "how do i change your voice" in cmd or "change your voice" in cmd or "change voice" in cmd or "voice change in cmd" in cmd:
                speak(" i am sorry about it , but my voice is fine")
                
        elif 'are you married' in cmd or 'do you have a girlfriend' in cmd:
            speak(" i am married to my work")
            
        elif "can you get me to instagram" in cmd or "i want to make a post in my instagram account" in cmd or "sam take me to instagram" in cmd or "can you please take me to instagram" in cmd:
            speak("OK, taking you to instagram.com")
            speak("Please wait for seconds")
            webbrowser.open_new_tab("https://www.instagram.com/")
            
        elif "can you take me to facebook" in cmd or "i want to make a status in facebook" in cmd or "sam take me to facebook" in cmd :
            speak("opening facebook.com")
            speak("Please wait for seconds")
            
            webbrowser.open_new_tab("https://www.facebook.com/")
        elif "can you recommend the best programming language to be learnt" in cmd or "can you guide me which is the best programming language" in cmd or "which is the best programming language " in cmd or "Trending programming languages" in cmd:
            speak("i will recommend you to learn Python as it is the most user friendly and easiest programming language which gives more job oppurtunities")
            speak("or else you can learn java script, swift, java, c, c++ or php")

        elif"search youtube" in cmd or "play a  song on youtube"in cmd or "play a song in youtube" in cmd or"search youtube" in cmd or "search for a video in youtube" in cmd or "search a video"  in cmd or "play a video on youtube" in cmd:
            speak("alright what song do i need to search for")
            song = Command()
            webbrowser.open("https://www.youtube.com/results?search_query="+song)

            
        elif"play music" in cmd or "i want to hear some music" in cmd or "play some songs" in cmd or "play a song" in cmd:
            webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            
        elif "open whatsapp" in cmd or  "whatsapp" in cmd:
            speak("alright opening whatsapp")
            webbrowser.open("https://web.whatsapp.com")
            
        elif "open notepad" in cmd or "notepad" in cmd or "please open notepad" in cmd :
                subprocess.Popen("c:\\Windows\\System32\\notepad.exe")


        elif "open wordpad" in cmd or"wordpad"  in cmd or "please open wordpad" in cmd:
            subprocess.Popen("c:\\Windows\\System32\\wordopenpad.exe")
            
        elif "tell me a joke" in cmd or  "please tell me a joke" in cmd or  "crack a joke" in cmd:
            joke = ["Hey Rachyl, do you remember me? Person 2: Wrong number. Person 1: What’s your number then?","Mom: How make chicken Daughter: What? Mom: Where buy chicken Daughter: Mom, this isn’t Google. Mom: Avocado"]
            speak(random.choice(joke))

        elif 'open stackoverflow' in cmd:
            speak("Here you go to Stack Over flow.Happy coding")
            webbrowser.open("stackoverflow.com")

        elif "what's your name" in cmd or "What is your name" in cmd:
            speak("My friends call me")
            speak("sam")
        
            

            
        elif "who i am" in cmd or "who am i" in cmd:
            speak("Since you talk In English, you must be an human I guess ?")
            
        elif "why you came to world" in cmd:
            speak("Thanks to those who invented me , further it is a secret")
            


        elif 'shutdown system' in cmd or "shut down my laptop" in cmd :
                speak("Hold On a Sec ! Your system is on its way to shut down")
                subprocess.call("shutdown /s /t 1")



        elif "don't listen" in cmd or "stop listening" in cmd or "pause" in cmd :
            speak("for how much time you want to stop sam from listening commands")
            a = int(Command())
            time.sleep(a)
            print(a)
            

            




        elif "write a note" in cmd:
            speak("What should i write, sir")
            note = Command()
            file = open('sam.txt', 'w')
            speak("Sir, Should i include date and time")
            file.write(note)
            speak("Alrigh, Note created")

        elif "show note" in cmd:
            speak("Showing Notes")
            file = open("sam.txt", "r")
            print(file.read())
            speak(file.read(6))
            
        elif "take me to twitter" in cmd or "can you take me to twitter" in cmd or "please get me to twitter" in cmd :
            speak("Ok taking you to twitter.com")
            webbrowser.open_new_tab("https://twitter.com/login?lang=en")
            
        elif "please take me to google" in cmd or "can you take me to google" in cmd or "i want to browse in google" in cmd:
            speak("Taking you to google.com")
            webbrowser.open_new_tab("https://www.google.com/")
            
        elif "move forward" in cmd or "go forward" in cmd:
            speak("Command Recognized , Moving forward")
            webbrowser.open("192.168.137.115/bt")
            
        elif "move backward" in cmd or "go backward" in cmd:
            speak("Command Recognized , Moving Backward")
            webbrowser.open("192.168.137.115/ft")   
            
        elif "turn towards left" in cmd or "turn left" in cmd or "move left" in cmd:
            speak("command recognized,moving towards left")
            webbrowser.open("192.168.137.115/left")
            
        elif "turn towards right" in cmd or "turn right" in cmd or "move right" in cmd:
            speak("command recognized, moving towards right")
            webbrowser.open("192.168.137.115/right")
        elif "handshake" in cmd:
            speak("Hello User")
            webbrowser.open("192.168.137.230/inline")
        
        elif "rotate" in cmd:
            speak("command recognized , rotating")
            webbrowser.open("192.168.137.115/rt")

            

        elif "open command prompt" in cmd or "open commandprompt" in cmd or "open cmd" in cmd or "cmd" in cmd or "command prompt" in cmd:
            subprocess.call('cmd.exe')
            

        elif "who is yashwanth k" in cmd:
            speak("he is gay")
                                                
        
        else:
            cmd = cmd
            try:
                try:
                    res = client.query(cmd)
                    results = next(res.results).text
                    speak(results)
                    
                except:
                      ans =Reply(cmd)
                      speak(ans)
        
            except:
                speak("whoops, that wasnt supposed to happen")
                speak("I am  not quite sure about that, anything else I can help you with ?")
        
        speak('any other commands?')
