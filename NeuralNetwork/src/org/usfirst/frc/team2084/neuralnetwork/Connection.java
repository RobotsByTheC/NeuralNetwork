/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

/**
 * Represents a connection between two {@link Neuron}s, and holds a weight and
 * its delta.
 * 
 * @author Ben Wolsieffer
 */
public class Connection {

    /**
     * The weight of this connection.
     */
    public double weight = Math.random();
    /**
     * The change in the weight from the last time it was updated. This is used
     * for momentum calculation.
     */
    public double deltaWeight = 0;
    
    public void reset() {
        weight = Math.random();
        deltaWeight = 0;
    }
}
