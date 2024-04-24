#include <Wire.h>
#include <SPI.h>
#include <Adafruit_LSM9DS1.h>
#include <Adafruit_Sensor.h>  // not used in this demo but required!
#include <SimpleKalmanFilter.h>
#include <Adafruit_PWMServoDriver.h>

SimpleKalmanFilter roll_k(1, 1, 0.5);
SimpleKalmanFilter pitch_k(1, 1, 0.075);

SimpleKalmanFilter index_k(1, 1, 0.005);
SimpleKalmanFilter middle_k(1, 1, 0.05);
SimpleKalmanFilter ring_k(1, 1, 0.05);
SimpleKalmanFilter little_k(1, 1, 0.05);

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();


// i2c
Adafruit_LSM9DS1 lsm = Adafruit_LSM9DS1();

#define LSM9DS1_SCK A5
#define LSM9DS1_MISO 12
#define LSM9DS1_MOSI A4
#define LSM9DS1_XGCS 6
#define LSM9DS1_MCS 5

#define SERVOMIN  150 // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  600 // This is the 'maximum' pulse length count (out of 4096)
#define USMIN  600 // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
#define USMAX  2400 // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates


float r;
float p;
float y;
float KalmanAngleRoll=0; 
float KalmanAnglePitch=0;
float RateRoll, RatePitch, RateYaw;
float RateCalibrationRoll, RateCalibrationPitch, RateCalibrationYaw; 
float AngleCalibrationPitch=1.84, AngleCalibrationRoll=-0.58;
int RateCalibrationNumber;
float AccX, AccY, AccZ;
float AngleRoll, AnglePitch;

float index_angle = 0,
    middle_angle = 0,
    ring_angle = 0,
    little_angle = 0;

// EMA Smoothing factors
float alpha = 0.08;

// EMA states
float emaIndex = 0;
float emaMiddle = 0;
float emaRing = 0;
float emaLittle = 0;

uint8_t servonum = 0;

void setupSensor()
{
  //Set the accelerometer range
  lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_2G, lsm.LSM9DS1_ACCELDATARATE_10HZ);
}

void gyro_signals(void) {
  lsm.read();  /* ask it to read in the data */ 

  /* Get a new sensor event */ 
  sensors_event_t a, m, g, temp;

  lsm.getEvent(&a, &m, &g, &temp); 
  RateRoll = g.gyro.x;
  RatePitch = g.gyro.y;
  RateYaw = g.gyro.z;
  AccX= a.acceleration.x;
  AccY= a.acceleration.y ;
  AccZ= a.acceleration.z;
//  AngleRoll = atan(AccY / sqrt(pow(AccX, 2)+pow(AccZ, 2))) * 180/PI;
//  AnglePitch = -atan(AccX / sqrt(pow(AccY,2)+pow(AccZ,2))) * 180/PI;

// Updated the formulas for gravity being in y axis (Rotated IMU placed on bicep)
  AngleRoll = (atan(AccZ / sqrt(pow(AccX, 2)+pow(AccY, 2))) * 180/PI) ;
  AnglePitch = (-atan(AccX / sqrt(pow(AccY,2)+pow(AccZ,2))) * 180/PI) ;
}


void write(int servo_num, int angle) {
  
  int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  // Serial.println(pulse);
  pwm.setPWM(servo_num, 0, pulse);
}

void setup() 
{
  Serial.begin(57600);

//  while (!Serial) {
//    delay(1); // will pause Zero, Leonardo, etc until serial console opens
//  }
//  
//  Serial.println("LSM9DS1 data read demo");
//  
//  // Try to initialise and warn if we couldn't detect the chip
//  if (!lsm.begin())
//  {
//    Serial.println("Oops ... unable to initialize the LSM9DS1. Check your wiring!");
//    while (1);
//  }
//  Serial.println("Found LSM9DS1 9DOF");
//
//  // helper to just set the default scaling we want, see above!
//  setupSensor();

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates
  delay(10);
}

float mapfloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}


void loop() 
{

  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n'); // Read the incoming data until newline
    
    // Assuming the data format is "thumb,index,middle,ring,little\n"
    int firstCommaIndex = data.indexOf(',');
    float thumb = data.substring(0, firstCommaIndex).toFloat();
    
    int secondCommaIndex = data.indexOf(',', firstCommaIndex + 1);
    float index = data.substring(firstCommaIndex + 1, secondCommaIndex).toFloat();

    int thirdCommaIndex = data.indexOf(',', secondCommaIndex + 1);
    float middle = data.substring(secondCommaIndex + 1, thirdCommaIndex).toFloat();

    int fourthCommaIndex = data.indexOf(',', thirdCommaIndex + 1);
    float ring = data.substring(thirdCommaIndex + 1, fourthCommaIndex).toFloat();
    
    float little = data.substring(fourthCommaIndex + 1).toFloat();

    // Update EMA values
    emaIndex = alpha * index + (1 - alpha) * emaIndex;
    emaMiddle = alpha * middle + (1 - alpha) * emaMiddle;
    emaRing = alpha * ring + (1 - alpha) * emaRing;
    emaLittle = alpha * little + (1 - alpha) * emaLittle;

    // Map the filtered values to servo angles
    index_angle = mapfloat(emaIndex, 0.96, 2.1, 50, 140);
    middle_angle = mapfloat(emaMiddle, 0.76, 2.2, 20, 110);
    ring_angle = mapfloat(emaRing, 0.67, 2.1, 10, 100);
    little_angle = mapfloat(emaLittle, 0.72, 1.88, 40, 130);

    // Update servo positions
    write(8, index_angle); // Update index servo as an example
    write(0, index_angle); 
    write(1, middle_angle); 
    write(2, ring_angle); 
    write(3, little_angle); 
    
  String output = String(index_angle) + "," + String(middle_angle) + "," + 
                  String(ring_angle) + "," + String(little_angle);
  Serial.println(output);               
  
  }
//low end 70 diff
//write(3,30);
//write(2,30);
//write(1,40);
//write(0,70);


// 20 diff
//write(3,45);
//write(2,45);
//write(1,55);
//write(0, 85);

//diff mid
//write(3,65);   // 35
//write(2,65);   // 35
//write(1,75);   // 35
//write(0, 105); // 35

//  quater 20 diff
//write(3,80);
//write(2,80);
//write(1,90);
//write(0,120);


// Base values 
//write(3,100);
//write(2,100);
//write(1,110);
//write(0, 140);




//write(8, 120);

//  for ( int i = 0; i<120; i++){
//    write(0, i);
////    write(1, i);
////    write(2, i);
////    write(3, i);
//    delay(25);
//    }
//  for ( int i = 120; i>0; i--){
//    write(0, i);
////    write(1, i);
////    write(2, i);
////    write(3, i);
//    delay(25);
//    }

    //3 is lil finger min is 120  around 90/83 is mid 35
//2 is ring finger min is 110  around 75 is mid 35
//1 is middle finger min is 110  around 75 is mid 35
//0 is index finger min is 140  around 95 is mid 45

//3 is lil finger min is 100  around 90/83 is mid 35
//2 is ring finger min is 110  around 75 is mid 35
//1 is middle finger min is 110  around 75 is mid 35
//0 is index finger min is 140  around 95 is mid 45

// closed values
//write(3,55);
//write(2,40);
//write(1,40);
//write(0, 50); 

//// Middle values
//write(3,55);
//write(2,65);
//write(1,75);
//write(0, 95); 

//write(3,55);
//write(2,65);
//write(1,85);
//write(0, 105); 
}
