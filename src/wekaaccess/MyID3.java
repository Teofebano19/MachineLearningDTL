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
 * @author Teofebano, Andrey
 */
public class MyID3 extends Classifier{
    // Attribute
    private MyID3[] child;
    private Attribute attrSeparator;
    private double[] result;
    private double classValue;
    private Attribute classAttribute;
    
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
        Vector<Double> classCounter = new Vector<Double>();
        classCounter.setSize(data.numClasses());
        for (int i=0;i<classCounter.size();i++){
            classCounter.setElementAt(Double.valueOf(0), i);
        }
        int numInstance = data.numInstances();
        for (int i=0;i<numInstance;i++){
            int cv = (int) data.instance(i).classValue();
            classCounter.setElementAt(classCounter.elementAt(cv)+1, cv);
        }
        for (int i=0;i<data.numClasses();i++){
            if (classCounter.elementAt(i)>0){
                entropy -= classCounter.elementAt(i) * Utils.log2(classCounter.elementAt(i));
            }
        }
        entropy /= (double) data.numInstances();
        return entropy;
    }
    
    private Vector<Instances> split(Instances data, Attribute attr){
        Vector<Instances> group = new Vector<Instances>(attr.numValues());
        for (int i = 0; i < attr.numValues(); i++) {
            group.add(new Instances(data, data.numInstances()));
        }
        for (int i=0;i<data.numInstances();i++){
            int av = (int) data.instance(i).value(attr);
            group.elementAt(av).add(data.instance(i));
        }
        for (int i=0;i<group.size();i++){
            group.elementAt(i).compactify();
        }
        return group;
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities C = super.getCapabilities();
        C.disableAll();
        
        C.enable(Capability.NOMINAL_ATTRIBUTES);
        C.enable(Capability.NUMERIC_ATTRIBUTES);
        C.enable(Capability.NOMINAL_CLASS);
        C.enable(Capability.MISSING_CLASS_VALUES);
        C.setMinimumNumberInstances(0);

        return C;
    }
    
    private void makeTree(Instances trainingData) {
        if (trainingData.numInstances() == 0){
            attrSeparator = null;
            result = new double[trainingData.numClasses()];
            classValue = Instance.missingValue();
            return;
        }
        
        Vector<Double> listIG =  new Vector<Double>();
        listIG.setSize(trainingData.numAttributes());
        for (int i=0;i<listIG.size();i++){
            listIG.setElementAt(Double.valueOf(0), i);
        }
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
            result = new double[trainingData.numClasses()];
            for (int i=0;i<result.length;i++){
                result[(int)trainingData.instance(i).classValue()]++;
            }
            Utils.normalize(result);
            classValue = Utils.maxIndex(result);
            classAttribute = trainingData.classAttribute();
        }
        else{
            Vector<Instances> newData = split(trainingData,attrSeparator);
            child = new MyID3[attrSeparator.numValues()];
            for (int i=0;i<child.length;i++){
                child[i] = new MyID3();
                child[i].makeTree(newData.elementAt(i));
            }
        }
    }
    
//    private void makeClassifier(Instances trainingData) throws Exception{
//        getCapabilities().testWithFail(trainingData);
//        trainingData.deleteWithMissingClass();
//        makeTree(trainingData);
//    }
    
    @Override
    public double classifyInstance(Instance testingData) throws NoSupportForMissingValuesException, Exception{
        if (testingData.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyID3 can't handle such missing value");
        }
        if (attrSeparator == null){
            return classValue;
        }
        else{
            return child[(int) testingData.value(attrSeparator)].classifyInstance(testingData);
        }
    }    
    
    @Override
    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
          throw new NoSupportForMissingValuesException("MyID3 can't handle such missing value");
        }
        if (attrSeparator == null) {
            return result;
        } else { 
            return child[(int) instance.value(attrSeparator)].distributionForInstance(instance);
        }
    }
}
