/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

import java.io.File;
import java.util.Arrays;

/**
 * @author ben
 */
public class FileTest {

    public static final int MAX_EPOCHS = 1000000;
    public static final double MAX_ERROR = 0.01;

    /**
     * Trains the network using a standard input file. This is not that useful
     * for an FRC robot.
     */
    public void run() {
        try {
            Data data = new Data(new File("data/not.txt"));
            Network network = data.getNetwork();

            double[][] inputs = data.getInputs();
            double[][] targetOutputs = data.getTargetOutputs();

            System.out.println(Arrays.deepToString(inputs));
            System.out.println(Arrays.deepToString(targetOutputs));

            int epochs = 0;
            double averageError = 0;
            while (true) {
                double error = 0;

                epochs++;
                for (int i = 0; i < inputs.length; i++) {
                    double[] input = inputs[i];
                    double[] targetOutput = targetOutputs[i];
                    network.feedForward(input);
                    network.backPropagation(targetOutput);
                    error = network.getRecentAverageError();
                }

                averageError = error;
                if (epochs % 1000 == 0) {
                    System.out.println("Error: " + averageError);
                }

                if (averageError < MAX_ERROR) {
                    System.out.println("==============");
                    System.out.println("Took " + epochs + " epochs to converge.");
                    for (int i = 0; i < inputs.length; i++) {
                        double[] input = inputs[i];
                        network.feedForward(input);
                        System.out.println("Inputs: " + Arrays.toString(input));
                        System.out.println("Outputs: " + Arrays.toString(network.getResults()));
                    }

                    break;
                } else if (epochs >= MAX_EPOCHS) {
                    System.out.println("==============");
                    System.out.println("Did not converge after " + epochs + " epochs.");
                    break;
                }
            }
            // Save the network to an output file
            data.save(new File("data/out.txt"));
            // network.feedForward(new double[] { 1, 1 });
            // System.out.println(Arrays.toString(network.getResults()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
