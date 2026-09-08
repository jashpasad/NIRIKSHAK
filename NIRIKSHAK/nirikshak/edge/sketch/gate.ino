/*
 * NIRIKSHAK Tier-0 MCU sketch -- Arduino UNO Q (STM32U585, Cortex-M33 / Zephyr)
 * Jash Pasad, IIT Gandhinagar -- https://github.com/jashpasad/NIRIKSHAK
 *
 * The Linux side decides. This side acts, on time, every time.
 *
 * Splitting it this way is the reason the UNO Q is in this system rather than a
 * plain USB camera. A reject decision that arrives 30 ms late has diverted the
 * wrong part, and 30 ms of jitter is an ordinary Tuesday for a Linux userspace
 * process. The Cortex-M33 running Zephyr has none of that, so the actuation
 * window is deterministic while the perception stays on the big cores.
 */

#include <Arduino.h>

const int PIN_DIVERTER   = 9;   // solenoid driver
const int PIN_LAMP_GREEN = 6;
const int PIN_LAMP_RED   = 5;
const int PIN_TRIGGER_IN = 2;   // optional hardware part-present sensor

volatile unsigned long diverterUntil = 0;
volatile unsigned long partCount     = 0;

void fireDiverter(unsigned int durationMs) {
  digitalWrite(PIN_DIVERTER, HIGH);
  diverterUntil = millis() + durationMs;
}

void onPartSensor() { partCount++; }

void setup() {
  pinMode(PIN_DIVERTER, OUTPUT);
  pinMode(PIN_LAMP_GREEN, OUTPUT);
  pinMode(PIN_LAMP_RED, OUTPUT);
  pinMode(PIN_TRIGGER_IN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_TRIGGER_IN), onPartSensor, FALLING);

  Serial.begin(115200);
  digitalWrite(PIN_LAMP_GREEN, HIGH);
}

/*
 * Commands from the Linux side, newline terminated:
 *   R<ms>   reject: fire the diverter for <ms>
 *   G       green lamp  (pass)
 *   A       amber/red   (review or fail, no divert)
 *   C       report part count
 */
void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("R")) {
      fireDiverter(cmd.substring(1).toInt());
      digitalWrite(PIN_LAMP_GREEN, LOW);
      digitalWrite(PIN_LAMP_RED, HIGH);
    } else if (cmd == "G") {
      digitalWrite(PIN_LAMP_GREEN, HIGH);
      digitalWrite(PIN_LAMP_RED, LOW);
    } else if (cmd == "A") {
      digitalWrite(PIN_LAMP_GREEN, LOW);
      digitalWrite(PIN_LAMP_RED, HIGH);
    } else if (cmd == "C") {
      Serial.print("count:"); Serial.println(partCount);
    }
  }

  // Non-blocking release. delay() here would stall the command loop and is the
  // single most common way this kind of sketch drops a reject.
  if (diverterUntil && millis() > diverterUntil) {
    digitalWrite(PIN_DIVERTER, LOW);
    diverterUntil = 0;
  }
}
