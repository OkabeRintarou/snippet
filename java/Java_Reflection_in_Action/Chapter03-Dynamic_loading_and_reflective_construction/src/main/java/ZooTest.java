import org.jdom2.Document;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class ZooTest {
    private static final String text = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" +
            "<serialized>\n" +
            "    <object class=\"Zoo\" id=\"0\">\n" +
            "        <field name=\"city\" declaringclass=\"Zoo\">\n" +
            "            <reference>1</reference>\n" +
            "        </field>\n" +
            "        <field name=\"name\" declaringclass=\"Zoo\">\n" +
            "            <reference>3</reference>\n" +
            "        </field>\n" +
            "        <field name=\"animals\" declaringclass=\"Zoo\">\n" +
            "            <reference>5</reference>\n" +
            "        </field>\n" +
            "    </object>\n" +
            "    <object class=\"java.lang.String\" id=\"1\">\n" +
            "        <field name=\"value\" declaringclass=\"java.lang.String\">\n" +
            "            <reference>2</reference>\n" +
            "        </field>\n" +
            "        <field name=\"hash\" declaringclass=\"java.lang.String\">\n" +
            "            <value>0</value>\n" +
            "        </field>\n" +
            "    </object>\n" +
            "    <object class=\"[C\" id=\"2\" length=\"24\">\n" +
            "        <value>N</value>\n" +
            "        <value>a</value>\n" +
            "        <value>t</value>\n" +
            "        <value>i</value>\n" +
            "        <value>o</value>\n" +
            "        <value>n</value>\n" +
            "        <value>a</value>\n" +
            "        <value>l</value>\n" +
            "        <value />\n" +
            "        <value>Z</value>\n" +
            "        <value>o</value>\n" +
            "        <value>o</value>\n" +
            "        <value>l</value>\n" +
            "        <value>o</value>\n" +
            "        <value>g</value>\n" +
            "        <value>i</value>\n" +
            "        <value>c</value>\n" +
            "        <value>a</value>\n" +
            "        <value>l</value>\n" +
            "        <value />\n" +
            "        <value>P</value>\n" +
            "        <value>a</value>\n" +
            "        <value>r</value>\n" +
            "        <value>k</value>\n" +
            "    </object>\n" +
            "    <object class=\"java.lang.String\" id=\"3\">\n" +
            "        <field name=\"value\" declaringclass=\"java.lang.String\">\n" +
            "            <reference>4</reference>\n" +
            "        </field>\n" +
            "        <field name=\"hash\" declaringclass=\"java.lang.String\">\n" +
            "            <value>0</value>\n" +
            "        </field>\n" +
            "    </object>\n" +
            "    <object class=\"[C\" id=\"4\" length=\"16\">\n" +
            "        <value>W</value>\n" +
            "        <value>a</value>\n" +
            "        <value>s</value>\n" +
            "        <value>h</value>\n" +
            "        <value>i</value>\n" +
            "        <value>n</value>\n" +
            "        <value>g</value>\n" +
            "        <value>t</value>\n" +
            "        <value>o</value>\n" +
            "        <value>n</value>\n" +
            "        <value>,</value>\n" +
            "        <value />\n" +
            "        <value>D</value>\n" +
            "        <value>.</value>\n" +
            "        <value>C</value>\n" +
            "        <value>.</value>\n" +
            "    </object>\n" +
            "    <object class=\"java.util.ArrayList\" id=\"5\">\n" +
            "        <field name=\"elementData\" declaringclass=\"java.util.ArrayList\">\n" +
            "            <null />\n" +
            "        </field>\n" +
            "        <field name=\"size\" declaringclass=\"java.util.ArrayList\">\n" +
            "            <value>2</value>\n" +
            "        </field>\n" +
            "        <field name=\"modCount\" declaringclass=\"java.util.AbstractList\">\n" +
            "            <null />\n" +
            "        </field>\n" +
            "    </object>\n" +
            "</serialized>";

    public static void main(String[] args) throws Exception {
        Document doc = null;

        try {
            SAXBuilder builder = new SAXBuilder();
            InputStream stream = new ByteArrayInputStream(text.getBytes());
            doc = builder.build(stream);
        } catch (JDOMException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


        Zoo zoo = (Zoo) Driver.deserializeObject(doc);
        System.out.println(zoo);

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

    public Zoo() {

    }

    public Zoo(String city, String name) {
        this.city = city;
        this.name = name;
        this.animals = new ArrayList();
    }

    public void add(Animal animal) {
        animals.add(animal);
    }

    @Override
    public String toString() {
        return "Zoo " + name + "@" + city;
    }
}
