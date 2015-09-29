/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaaccess;

import java.util.Collections;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.core.Capabilities.Capability;
import weka.core.*;

/**
 *
 * @author Teofebano
 */
public class MyID3 extends Classifier{
    // Attribute
    private Vector<MyID3> child;
    private Attribute attrSeparator;
    private Vector<Double> result;
    private double classValue;
    
    // Code
    public MyID3(){
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        // Missing Class
        data = new Instances(data);
        data.deleteWithMissingClass();

        makeTree(data);
    }
    
    private double computeIG(Instances data, Attribute attr){
       double IG = computeEntropy(data);
       Vector<Instances> instances = split(data,attr);
       for (int i=0;i<attr.numValues();i++){
           if (instances.elementAt(i).numInstances() > 0){
               IG -= (instances.elementAt(i).numInstances() / data.numInstances()) * computeEntropy(instances.elementAt(i));
           }
       }
       return IG;
    }
    
    private double computeEntropy(Instances data){
        double entropy = 0;
        Vector<Double> classCounter = new Vector<Double>(data.numClasses());
        int numInstance = data.numInstances();
        System.out.println(classCounter.elementAt(0));
        for (int i=0;i<numInstance;i++){
            int cv = (int) data.instance(i).classValue();
            System.out.println(cv);
            classCounter.setElementAt(classCounter.elementAt(cv)+1, cv);
        }
        for (int i=0;i<numInstance;i++){
            if (classCounter.elementAt(i)>0){
                entropy -= classCounter.elementAt(i) * Utils.log2(classCounter.elementAt(i));
            }
        }
        entropy /= (double) data.numInstances();
        return entropy;
    }
    
    private Vector<Instances> split(Instances data, Attribute attr){
        Vector<Instances> group = new Vector<Instances>(attr.numValues()); 
        for (int i=0;i<attr.numValues();i++){
            int av = (int) data.instance(i).value(attr);
            group.elementAt(av).add(data.instance(i));
        }
        for (int i=0;i<group.size();i++){
            group.elementAt(i).compactify();
        }
        return group;
    }
    
    public Capabilities getCapabilities() {
        Capabilities capa = super.getCapabilities();
        capa.disableAll();
        
        capa.enable(Capability.NOMINAL_ATTRIBUTES);
        capa.enable(Capability.NOMINAL_CLASS);
        capa.enable(Capability.MISSING_CLASS_VALUES);
        capa.setMinimumNumberInstances(0);

        return capa;
    }
    
    private void makeTree(Instances trainingData) {
        if (trainingData.numInstances() == 0){
            attrSeparator = null;
            result = new Vector<>(trainingData.numClasses());
            classValue = Instance.missingValue();
            return;
        }
        
        Vector<Double> listIG =  new Vector<Double>(trainingData.numAttributes());
        for (int i=0;i<trainingData.numAttributes();i++){
            Attribute attr = trainingData.attribute(i);
            int attrIndex = attr.index();
            listIG.setElementAt(computeIG(trainingData, attr), attrIndex);
        }
        int index = listIG.indexOf(Collections.max(listIG));
        attrSeparator = trainingData.attribute(index);
        
        // Build Tree
        if (listIG.elementAt(index) == 0){
            attrSeparator = null;
            result = new Vector<Double>(trainingData.numClasses());
            for (int i=0;i<trainingData.numInstances();i++){
                int cv = (int) trainingData.instance(i).classValue();
                result.setElementAt(result.elementAt(cv)+1, cv);
            }
            // Change Vector to Double
            double[] arrResult = new double[result.size()];
            for (int i=0;i<arrResult.length;i++){
                arrResult[i] = result.elementAt(i);
            }
            Utils.normalize(arrResult);
            classValue = Utils.maxIndex(arrResult);
        }
        else{
            Vector<Instances> newData = split(trainingData,attrSeparator);
            child = new Vector<>(attrSeparator.numValues());
            for (int i=0;i<child.size();i++){
                child.elementAt(i).makeTree(newData.elementAt(i));
            }
        }
    }
    
    private void makeClassifier(Instances trainingData) throws Exception{
        getCapabilities().testWithFail(trainingData);
        trainingData.deleteWithMissingClass();
        makeTree(trainingData);
    }
    
   
    
    @Override
    public double classifyInstance(Instance testingData) throws NoSupportForMissingValuesException, Exception{
        if (testingData.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyID3 can't handle such missing value");
        }
        if (attrSeparator == null){
            return classValue;
        }
        else{
            return child.elementAt((int) testingData.value(attrSeparator)).classifyInstance(testingData);
        }
    }    
    
    @Override
    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
          throw new NoSupportForMissingValuesException("MyID3 can't handle such missing value");
        }
        double[] arrResult = new double[result.size()];
        if (attrSeparator == null) {
            for (int i=0;i<result.size();i++){
                arrResult[i] = result.elementAt(i);
            }
            return arrResult;
        } else { 
            return child.elementAt((int) instance.value(attrSeparator)).distributionForInstance(instance);
        }
    }
}
