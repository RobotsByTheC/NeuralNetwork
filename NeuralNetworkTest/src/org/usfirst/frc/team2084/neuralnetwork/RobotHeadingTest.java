/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

import java.awt.Container;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CompassPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.general.DefaultValueDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

/**
 * @author ben
 */
public class RobotHeadingTest implements Test {

    public static final double MAX_ERROR = 5;

    /**
     * A robot "simulation", used to test the unsupervised learning ability.
     */
    private static class Robot {

        public static final double MAX_ACCELERATION = 5;
        public static final double TIME_STEP = 0.05;
        public static final double MAX_SPEED = 50;

        public double speed = 0;
        public double heading = 0;

        public void rotate(double power) {
            speed += power * TIME_STEP * MAX_ACCELERATION;
            if (speed > MAX_SPEED) {
                speed = MAX_SPEED;
            } else if (speed < -MAX_SPEED) {
                speed = -MAX_SPEED;
            }
            heading += speed * TIME_STEP * MAX_SPEED;
        }
    }

    private volatile boolean running = false;
    private volatile Thread thread;

    /**
     * 
     */
    @Override
    public void run() {
        try {
            final DefaultValueDataset headingData = new DefaultValueDataset(0);
            final DefaultValueDataset desiredHeadingData = new DefaultValueDataset(0);
            final CompassPlot headingPlot = new CompassPlot();
            headingPlot.addDataset(headingData);
            headingPlot.addDataset(desiredHeadingData);
            final JFreeChart headingChart = new JFreeChart("Heading", headingPlot);

            final XYSeries headingTimeSeries = new XYSeries("Heading");
            final XYSeriesCollection headingTimeData = new XYSeriesCollection();
            headingTimeData.addSeries(headingTimeSeries);
            final JFreeChart headingTimeChart = ChartFactory.createXYLineChart("Heading vs. Time", "Time", "Heading", headingTimeData, PlotOrientation.VERTICAL, true, true, false);

            final XYSeries errorTimeSeries = new XYSeries("Error");
            final XYSeriesCollection errorTimeData = new XYSeriesCollection();
            errorTimeData.addSeries(errorTimeSeries);
            final JFreeChart errorTimeChart = ChartFactory.createXYLineChart("Error vs. Time", "Time", "Error", errorTimeData, PlotOrientation.VERTICAL, true, true, false);

            SwingUtilities.invokeAndWait(() -> {
                final JFrame frame = new JFrame("Charts");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                final Container content = frame.getContentPane();
                content.setLayout(new BoxLayout(content, BoxLayout.PAGE_AXIS));

                final JPanel chartPanel = new JPanel();
                chartPanel.setLayout(new GridLayout(2, 2));
                content.add(chartPanel);

                final ChartPanel headingPanel = new ChartPanel(headingChart);
                chartPanel.add(headingPanel);

                final ChartPanel headingTimePanel = new ChartPanel(headingTimeChart);
                chartPanel.add(headingTimePanel);

                final ChartPanel errorTimePanel = new ChartPanel(errorTimeChart);
                chartPanel.add(errorTimePanel);

                final JPanel buttonPanel = new JPanel();
                content.add(buttonPanel);

                final JButton startButton = new JButton("Start");
                final JButton stopButton = new JButton("Stop");

                startButton.addActionListener(new ActionListener() {

                    @Override
                    public void actionPerformed(ActionEvent e) {
                        stop();
                        startButton.setEnabled(false);
                        stopButton.setEnabled(true);
                        start(headingData, desiredHeadingData, headingTimeSeries, errorTimeSeries);
                    }
                });
                buttonPanel.add(startButton);

                stopButton.addActionListener(new ActionListener() {

                    @Override
                    public void actionPerformed(ActionEvent e) {
                        stop();
                        startButton.setEnabled(true);
                        stopButton.setEnabled(false);
                    }
                });
                stopButton.setEnabled(false);
                buttonPanel.add(stopButton);

                frame.pack();
                frame.setVisible(true);
            });
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private void start(DefaultValueDataset headingData, DefaultValueDataset desiredHeadingData, XYSeries headingTimeSeries, XYSeries errorTimeSeries) {
        try {
            thread = new Thread(new Runnable() {

                @Override
                public void run() {
                    try {

                        Robot robot = new Robot();
                        // Load the robot network characteristics from a file
                        Data data = new Data(new File("data/robot.txt"));
                        Network network = data.getNetwork();

                        double time = 0;
                        double error = 1;
                        // Pick a random desired heading
                        double desired = Math.random();
                        System.out.println("Trying to rotate to heading: " + desired);
                        desiredHeadingData.setValue(desired);
                        headingTimeSeries.clear();
                        errorTimeSeries.clear();
                        do {
                            // Feed forward the heading error
                            network.feedForward(desired - robot.heading);
                            // Rotate the robot at the speed given by the
                            // output of last hidden layer
                            double speed = network.getLayer(network.getTotalLayers() - 2)[0].getOutputValue();
                            robot.rotate(speed);
                            // Apply that rotation speed for a certain
                            // amount of time
                            Thread.sleep((int) (Robot.TIME_STEP * 1000));
                            time += Robot.TIME_STEP;
                            // Set the output neuron to the new heading
                            // error. This is the kind of weird part,
                            // because it tricks the back-propagation
                            // algorithm into minimizing the difference
                            // between the real and desired headings.
                            network.setLayerOutputs(network.getTotalLayers() - 1, desired - robot.heading);
                            // Back-propagate to adjust weights to minimize
                            // error
                            network.backPropagation(0);
                            error = network.getRecentAverageError();
                            // System.out.println("Speed: " + speed);
                            // System.out.println("Heading: " +
                            // robot.heading);
                            headingData.setValue(robot.heading * 180);
                            headingTimeSeries.add(time, robot.heading);
                            errorTimeSeries.add(time, error);
                            // System.out.println("Error: " + error + "\n");
                        } while (running);
                        System.out.println("====================");
                        System.out.println("Success!");
                        System.out.println("Final error: " + error);
                        System.out.println("====================");
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                }
            });
            running = true;
            thread.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void stop() {
        running = false;
        if (thread != null) {
            try {
                thread.join();
            } catch (InterruptedException e) {
            }
        }
    }

    private static void pause() {
        System.out.println("Press any key to continue...");
        try {
            System.in.read();
        } catch (Exception e) {
        }
    }
}
