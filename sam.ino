#define BLYNK_PRINT Serial            
#include <BlynkSimpleEsp8266.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h>
#include <Servo.h>
#include <ESP8266NetBIOS.h>

char auth[] = "n-fW6aLmDNeR87JGQXiWqYqoXSBE1fGd";
 
char ssid[] = "FIITJEE_R4";  // type your wifi name
char pass[] = "IITJEE@17";    // type your wifi password


/* Lexin */
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;


BlynkTimer timer; 

/*Blynk is an open source application and webserver */



#define a D5
#define b D6
#define c D7
#define d D8

int servo1_pin = D1;
int servo2_pin = D2;
int servo3_pin = D3;
int servo4_pin = D4;

int servo1_positions[] = {90, 180, 90, 180};  // You can modify the positions as per your dance move
int servo2_positions[] = {120, 60, 120, 60};
int servo3_positions[] = {75, 105, 75, 105};
int servo4_positions[] = {135, 95, 135, 95};

int delay_time = 1000;

void greet(){

}

void DanceMove() {
  for (int i = 0; i < 3; i++) {
    servo1.write(servo1_positions[i]);
    servo2.write(servo2_positions[i]);
    servo3.write(servo3_positions[i]);
    servo4.write(servo4_positions[i]);
    delay(delay_time);
  }
}



void Forward() {
  digitalWrite(a, HIGH);
  digitalWrite(b, LOW);
  digitalWrite(c, LOW);
  digitalWrite(d, HIGH);
}

void Backward() {
  digitalWrite(a, LOW);
  digitalWrite(b, HIGH);
  digitalWrite(c, HIGH);
  digitalWrite(d, LOW); 
}


void Ft() {
  digitalWrite(a, HIGH);
  digitalWrite(b, LOW);
  digitalWrite(c, LOW);
  digitalWrite(d, HIGH);
  delay(1500);
  digitalWrite(a, LOW);
  digitalWrite(b, LOW);
  digitalWrite(c, LOW);
  digitalWrite(d, LOW);
}



void Bt() {
  digitalWrite(a, LOW);
  digitalWrite(b, HIGH);
  digitalWrite(c, HIGH);
  digitalWrite(d, LOW);
  delay(1500);
  digitalWrite(a, LOW);
  digitalWrite(b, LOW);
  digitalWrite(c, LOW);
  digitalWrite(d, LOW);
}



void Rt() {
  digitalWrite(a, LOW);
  digitalWrite(b, HIGH);
  digitalWrite(c, LOW);
  digitalWrite(d, LOW);
  delay(550);
  digitalWrite(a, LOW);
  digitalWrite(b, LOW);
  digitalWrite(c, LOW);
  digitalWrite(d, LOW);
}

  

void Lt() {
  digitalWrite(a, HIGH);  
  digitalWrite(b, LOW);
  digitalWrite(c, LOW);
  digitalWrite(d, LOW);
  delay(550);
  digitalWrite(a, LOW);
  digitalWrite(b, LOW);
  digitalWrite(c, LOW);
  digitalWrite(d, LOW);
}



void Rotate() {
  digitalWrite(a, HIGH);
  digitalWrite(b, LOW);
  digitalWrite(c, HIGH);
  digitalWrite(d, LOW);
}



void Stop() {
  digitalWrite(a, LOW);
  digitalWrite(b, LOW);
  digitalWrite(c, LOW);
  digitalWrite(d, LOW);
}

BLYNK_WRITE(V0) {
  digitalWrite(a, param.asInt());
}
BLYNK_WRITE(V1) {
  digitalWrite(b, param.asInt());
}

BLYNK_WRITE(V2) {
  digitalWrite(c, param.asInt());
}

BLYNK_WRITE(V3) {
  digitalWrite(d, param.asInt());
}





ESP8266WebServer server(80);




void handleRoot() {
  server.send(200, "text/plain", "hello from Lexin!");
}


void handleNotFound() {
  String message = "File Not Found\n\n";
  message += "URI: ";
  message += server.uri();
  message += "\nMethod: ";
  message += (server.method() == HTTP_GET) ? "GET" : "POST";
  message += "\nArguments: ";
  message += server.args();
  message += "\n";
  for (uint8_t i = 0; i < server.args(); i++) {
    message += " " + server.argName(i) + ": " + server.arg(i) + "\n";
  }
  server.send(404, "text/plain", "404 Error");
}


void setup(void) {
  pinMode(a, OUTPUT);
  pinMode(b, OUTPUT);
  pinMode(c, OUTPUT);
  pinMode(d, OUTPUT);


  servo1.attach(servo1_pin);
  servo2.attach(servo2_pin);
  servo3.attach(servo3_pin);
  servo4.attach(servo4_pin);


  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, pass);
  Serial.println("");
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");



    Serial.println("");

    Serial.print("Connected to ");

    Serial.println(ssid);

    Serial.print("IP address: ");

    Serial.println(WiFi.localIP());

    if (MDNS.begin("esp8266")) {

      Serial.println("MDNS responder started");
    }


    Blynk.begin(auth, ssid, pass, "blynk.cloud", 80);
  }
  



  server.on("/", handleRoot);


  server.on("/ft", []() {
    server.send(200, "text/plain", "Forward");
    Ft();
  });



  server.on("/bt", []() {
    server.send(200, "text/plain", "Backward");
    Bt();
  });


  server.on("/left", []() {
    server.send(200, "text/plain", "Left");
    Lt();
  });



  server.on("/right", []() {
    server.send(200, "text/plain", "Right");
    Rt();
  });

  server.on("/forward", []() {
    server.send(200, "text/plain", "Forwarding");
    Forward();
  });
  server.on("/backward", []() {
    server.send(200, "text/plain", "Backwarding");
    Backward();
  });

  server.on("/dance", []() {
    server.send(200, "text/plain", "Dance Move  ");
    DanceMove();
  });

   
  server.on("/stop", []() {
    server.send(200, "text/plain", "Stop");
    Stop();
  });

  server.on("/rt", []() {
    server.send(200, "text/plain", "rotate");
    Rotate();
  });


  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("HTTP server started");
}




void loop(void) {
  server.handleClient();
  MDNS.update();  
  Blynk.run();
}
