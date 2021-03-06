package wekaaccess;

import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Teofebano, Andrey
 */
public class WekaAccess {
    // Attribute
    private static final String SOURCE = "data/iris.arff";
    private static final String SOURCE_UNLABELED = "data/iris.unlabeled.arff";
    private static final int NUMBER_FOLD = 10;
    private static final int PERCENTAGE = 66;
    private static Instances data, unseendata;
    
    // Code
    // CTOR
    public WekaAccess(){
    
    }
    
    // Load file
    public static void loadFile(String source){
        try {
            data = DataSource.read(source);
            if (data.classIndex() == -1){
                data.setClassIndex(data.numAttributes()-1);
            }
        } catch (Exception ex) {
        }
    }
    
    // 10-fold
    public static void learn10fold(Instances trainingData, Classifier classifier){
        try {
            // Build and Eval
            Evaluation eval = new Evaluation(trainingData);
            eval.crossValidateModel(classifier, trainingData, NUMBER_FOLD, new Random(1));
            
            // Print
            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(WekaAccess.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // percentage split
    public static void learnFull(Instances trainingData, Classifier classifier){
        try {
            // Build
            Classifier cls = classifier;
            int trainSize = (int) Math.round(trainingData.numInstances() * PERCENTAGE/ 100);
            int testSize = trainingData.numInstances() - trainSize;
            Instances train = new Instances(trainingData, 0, trainSize);
            Instances test = new Instances(trainingData, trainSize, testSize);
            cls.buildClassifier(train);
            // Eval
            Evaluation eval = new Evaluation(trainingData);
            eval.evaluateModel(classifier, test);
            
            // Print
            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(WekaAccess.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // save model
    public static void saveModel(Instances trainingData, Classifier classifier, String file){
        try {
            Classifier cls = classifier;
            cls.buildClassifier(trainingData);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
                oos.writeObject(cls);
                oos.flush();
            }
        } catch (Exception ex) {
            Logger.getLogger(WekaAccess.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // load model
    public static void loadModel(String file){
        ObjectInputStream ois;
        try {
            Classifier cls = null;
            ois = new ObjectInputStream(new FileInputStream(file));
            cls = (Classifier) ois.readObject(); 
            ois.close();
        } catch (FileNotFoundException e){
        } catch (IOException ex) {
            Logger.getLogger(WekaAccess.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(WekaAccess.class.getName()).log(Level.SEVERE, null, ex);
        } 
    }
    
    // classification
    public static void classifyUsingModel(Classifier classifier, String file){
        try {
            Instances unlabeled = DataSource.read(file);
            unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

            Instances labeled = new Instances(unlabeled);
            // label instances
            for (int i = 0; i < unlabeled.numInstances(); i++) {
                double clsLabel = classifier.classifyInstance(unlabeled.instance(i));
                labeled.instance(i).setClassValue(clsLabel);
                System.out.println(labeled.lastInstance().toString());
            }
        } catch (Exception e) {
                // TODO Auto-generated catch block
        }
    }
    
    // attribute removal
    public static Instances removeAttribute(String file, String indices, Boolean invert ) throws Exception{
        Instances newdata;
        
        Remove remove = new Remove();
        remove.setAttributeIndices(indices);
        remove.setInvertSelection(invert.booleanValue());
        remove.setInputFormat(data);
        newdata = Filter.useFilter(data, remove);
        
        return newdata;
    }
    
    // filter resample
    public static Instances resample(Instances data) throws Exception{
        Instances newdata;
        
        Resample resample = new Resample();
        resample.setInputFormat(data);
        newdata = Filter.useFilter(data, resample);
        
        return newdata;
    }
    
    // main
    public static void main(String[] args) {
        Classifier DT = new J48();
        MyID3 mid3 = new MyID3();
        MyJ48 my48 = new MyJ48();
        
        loadFile(SOURCE);
        
        
        // 10 fold
//        learn10fold(data, DT);
//        learn10fold(data, mid3);
//        learn10fold(data, my48);
        
        // full training
        learnFull(data, my48);
//        learnFull(data, mid3);
//        learnFull(data, my48);
        
        // unseen data
        classifyUsingModel(my48,SOURCE_UNLABELED);
    }
}
