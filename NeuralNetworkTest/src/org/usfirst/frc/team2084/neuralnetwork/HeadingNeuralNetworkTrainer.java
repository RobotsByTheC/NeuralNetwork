/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.GridLayout;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Paths;
import java.text.NumberFormat;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.ProgressMonitor;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.Range;
import org.jfree.data.xy.XYDataItem;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

/**
 * @author Ben Wolsieffer
 */
public class HeadingNeuralNetworkTrainer {

    private final JFrame frame = new JFrame("Heading Neural Network Trainer");
    private final JFileChooser dataDirectoryChooser = new JFileChooser(getWorkingDirectory());
    private final JFileChooser saveFileChooser = new JFileChooser(getWorkingDirectory());

    private final JPanel graphPanel = new JPanel();
    private final JPanel controlPanel = new JPanel();

    private ChartPanel outputGraphPanel;
    private final JFreeChart outputGraph;
    private final XYSeries outputGraphNetworkSeries;
    private final XYSeries outputGraphDataSeries;

    private final NumberFormat numberFormat = NumberFormat.getNumberInstance();

    private final JFormattedTextField etaField = new JFormattedTextField(numberFormat);
    private final JFormattedTextField momentumField = new JFormattedTextField(numberFormat);

    private final JButton chooseDataButton = new JButton("Choose Data");
    private final JButton trainButton = new JButton("Train");
    private final JButton saveButton = new JButton("Save");

    private File dataDirectory;
    private final Network network;

    private boolean training;
    private volatile int iterations = 10000;
    private volatile double eta = 0.3;
    private volatile double momentum = 0.1;

    public HeadingNeuralNetworkTrainer() {
        outputGraphNetworkSeries = new XYSeries("Network Prediction");
        outputGraphDataSeries = new XYSeries("Error");
        final XYSeriesCollection data = new XYSeriesCollection();
        data.addSeries(outputGraphDataSeries);
        data.addSeries(outputGraphNetworkSeries);
        outputGraph = ChartFactory.createXYLineChart("Error vs. Output", "Time", "Error", data,
                PlotOrientation.VERTICAL, true, true, false);

        NumberAxis xAxis = new NumberAxis();
        xAxis.setRange(new Range(-Math.PI, Math.PI));
        xAxis.setAutoRange(false);
        NumberAxis yAxis = new NumberAxis();
        yAxis.setRange(new Range(-1, 1));

        XYPlot plot = (XYPlot) outputGraph.getPlot();
        plot.setDomainAxis(xAxis);
        plot.setRangeAxis(yAxis);

        network = new Network(new int[] { 1, 5, 1 }, eta, momentum, new TransferFunction.HyperbolicTangent());

        try {
            SwingUtilities.invokeAndWait(() -> {
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                final Container content = frame.getContentPane();
                content.setLayout(new BorderLayout());

                outputGraphPanel = new ChartPanel(outputGraph);
                outputGraphPanel.setDomainZoomable(false);
                outputGraphPanel.setRangeZoomable(false);
                graphPanel.add(outputGraphPanel);

                graphPanel.setLayout(new GridLayout(1, 1));
                content.add(graphPanel, BorderLayout.CENTER);
                {

                    JLabel etaLabel = new JLabel("Eta:");
                    etaLabel.setLabelFor(etaField);
                    etaField.setText(Double.toString(network.getEta()));
                    etaField.setColumns(5);
                    etaField.addPropertyChangeListener("value",
                            (evt) -> eta = ((Number) evt.getNewValue()).doubleValue());
                    controlPanel.add(etaLabel);
                    controlPanel.add(etaField);

                    JLabel momentumLabel = new JLabel("Momentum:");
                    momentumLabel.setLabelFor(etaField);
                    momentumField.setText(Double.toString(network.getMomentum()));
                    momentumField.setColumns(5);
                    momentumField.addPropertyChangeListener("value",
                            (evt) -> momentum = ((Number) evt.getNewValue()).doubleValue());
                    controlPanel.add(momentumLabel);
                    controlPanel.add(momentumField);
                }

                chooseDataButton.addActionListener((e) -> {
                    if (dataDirectoryChooser.showOpenDialog(frame) == JFileChooser.APPROVE_OPTION) {
                        dataDirectory = dataDirectoryChooser.getSelectedFile();
                        displayData();
                    }
                });
                controlPanel.add(chooseDataButton);

                trainButton.addActionListener((e) -> trainNetwork());
                controlPanel.add(trainButton);

                saveButton.addActionListener((e) -> {
                    saveNetwork();
                });
                controlPanel.add(saveButton);

                content.add(controlPanel, BorderLayout.SOUTH);

                dataDirectoryChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.pack();
                frame.setVisible(true);
            });
        } catch (InvocationTargetException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void displayData() {
        if (dataDirectory != null) {
            outputGraphDataSeries.clear();

            // Create a copy of the data directory for the worker thread
            final File dataDirectory = this.dataDirectory;
            new Thread(() -> {
                if (dataDirectory.exists() && dataDirectory.isDirectory()) {
                    for (File d : dataDirectory.listFiles()) {
                        if (d.exists() && d.isFile() && d.getName().endsWith(".csv")) {
                            try (BufferedReader dataReader = new BufferedReader(new FileReader(d))) {
                                String line;
                                while ((line = dataReader.readLine()) != null) {
                                    String[] data = line.split(",");
                                    if (data.length >= 2) {
                                        double error = Double.parseDouble(data[0]);
                                        double output = Double.parseDouble(data[1]);

                                        // Add to the data series on the UI
                                        // thread
                                        SwingUtilities.invokeLater(() -> {
                                            outputGraphDataSeries.add(error, output);
                                        });
                                    } else {
                                        System.err.println("Too few values in data line: " + line);
                                    }
                                }
                            } catch (FileNotFoundException e) {
                                System.err.println("Could not open data file: " + e.getMessage());
                            } catch (IOException e) {
                                System.err.println("Could not read data file: " + e.getMessage());
                            } catch (NumberFormatException e) {
                                System.err.println("Data is not a valid number: " + e.getMessage());
                            }
                        }
                    }
                }
            }).start();
        }
    }

    private void displayNetwork() {
        outputGraphNetworkSeries.clear();

        for (double angle = -Math.PI; angle <= Math.PI; angle += 0.05) {
            double scaledAngle = convertAngleToInput(angle);
            synchronized (this) {
                network.feedForward(scaledAngle);
                double fAngle = angle;
                double output = network.getOutputLayer()[0].getOutputValue();

                SwingUtilities.invokeLater(() -> {
                    outputGraphNetworkSeries.add(fAngle, output);
                });
            }
        }
    }

    private void trainNetwork() {
        if (!training) {

            // Set training flag and disable button
            training = true;
            trainButton.setEnabled(false);

            final ProgressMonitor progressMonitor = new ProgressMonitor(frame, "Training Network...", "", 0, 100);
            progressMonitor.setMillisToDecideToPopup(100);
            progressMonitor.setMillisToPopup(400);

            @SuppressWarnings("unchecked")
            final ArrayList<XYDataItem> data = new ArrayList<>(outputGraphDataSeries.getItems());

            final int maxProgress = iterations * data.size();

            final SwingWorker<Void, Void> trainingWorker = new SwingWorker<Void, Void>() {

                @Override
                protected Void doInBackground() throws Exception {
                    // Reset the neural network to default values
                    synchronized (this) {
                        network.reset();
                        network.setEta(eta);
                        network.setMomentum(momentum);
                    }

                    outer: for (int j = 0; j < iterations; j++) {
                        for (int i = 0; i < data.size(); i++) {
                            if (!isCancelled()) {
                                XYDataItem d = data.get(i);
                                double error = convertAngleToInput(d.getXValue());
                                double output = d.getYValue();
                                synchronized (this) {
                                    network.feedForward(error);
                                    network.backPropagation(output);
                                }
                                int jl = j;
                                int il = i;
                                int progress = (int) (((float) (data.size() * jl + il + 1)
                                        / maxProgress) * 100);
                                setProgress(progress);
                            } else {
                                break outer;
                            }
                        }
                    }

                    displayNetwork();
                    return null;
                }

                @Override
                protected void done() {
                    training = false;
                    trainButton.setEnabled(true);
                    progressMonitor.close();
                }
            };
            trainingWorker.addPropertyChangeListener(new PropertyChangeListener() {

                @Override
                public void propertyChange(PropertyChangeEvent evt) {
                    if (evt.getPropertyName().equals("progress")) {
                        progressMonitor.setProgress((int) evt.getNewValue());
                    }
                    if (progressMonitor.isCanceled()) {
                        trainingWorker.cancel(true);
                    }
                }
            });
            trainingWorker.execute();
        }
    }

    private void saveNetwork() {
        if (saveFileChooser.showSaveDialog(frame) == JFileChooser.APPROVE_OPTION) {
            File saveFile = saveFileChooser.getSelectedFile();

            new SwingWorker<Void, Void>() {

                @Override
                protected Void doInBackground() throws Exception {
                    synchronized (network) {
                        new Data(network).save(saveFile);
                    }

                    return null;
                }

                @Override
                protected void done() {
                }
            }.execute();
        }
    }

    private static File getWorkingDirectory() {
        return new File(Paths.get("").toAbsolutePath().toString());
    }

    private static double convertAngleToInput(double angle) {
        return angle / Math.PI;
    }
}
