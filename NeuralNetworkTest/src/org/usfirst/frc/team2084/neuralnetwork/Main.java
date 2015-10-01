/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

/**
 * @author ben
 */
public class Main {

    public static void main(String[] args) {
        Test test = new RobotHeadingTest();
        // Test test = new FileTest();

        test.run();
    }
}
