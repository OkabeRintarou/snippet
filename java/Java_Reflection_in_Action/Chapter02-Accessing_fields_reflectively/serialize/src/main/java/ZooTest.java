import org.jdom2.Document;
import org.jdom2.output.Format;
import org.jdom2.output.XMLOutputter;

import java.util.ArrayList;
import java.util.List;

public class ZooTest {
    public static void main(String[] args) {
        Animal panda = new Animal(
                "tian tian",
                "male",
                "Ailuropoda melanoleuca",
                271);
        Animal panda2 = new Animal(
                "Mei Xiang",
                "female",
                "Ailuropoda melanoleuca",
                221);
        Zoo national = new Zoo(
                "National Zoological Park",
                "Washington, D.C.");

        national.add(panda);
        national.add(panda2);

        try {
            Format format = Format.getPrettyFormat();
            format.setIndent("    ");
            XMLOutputter out = new XMLOutputter(format);
            Document document = Driver.serializeObject(national);
            out.output(document, System.out);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

class Animal {
    private String name;
    private String gender;
    private String classification;
    private int weight;

    public Animal(String name, String gender, String classification, int weight) {
        this.name = name;
        this.gender = gender;
        this.classification = classification;
        this.weight = weight;
    }

}

class Zoo {
    private String city;
    private String name;
    private List animals;

    public Zoo(String city, String name) {
        this.city = city;
        this.name = name;
        this.animals = new ArrayList();
    }

    public void add(Animal animal) {
        animals.add(animal);
    }
}
