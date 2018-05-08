import org.jdom2.Document;
import org.jdom2.Element;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.IdentityHashMap;
import java.util.Map;

public class Driver {
    public static Document serializeObject(Object source) throws Exception {
        return serializeHelper(source, new Document(new Element("serialized")),
                new IdentityHashMap());
    }

    private static Document serializeHelper(Object source,
                                            Document target,
                                            Map table)
            throws Exception {
        String id = Integer.toString(table.size());
        table.put(source, id);       // Creates a unique identifier for object to be serialized
        Class sourceClass = source.getClass();
        Element oElt = new Element("object"); // Creates an XML element for object
        oElt.setAttribute("class", sourceClass.getName());
        oElt.setAttribute("id", id);
        target.getRootElement().addContent(oElt);

        if (!sourceClass.isArray()) {
            Field[] fields = Mopex.getInstanceVariables(sourceClass);
            for (int i = 0; i < fields.length; i++) {
                if (!Modifier.isPublic(fields[i].getModifiers())) {
                    fields[i].setAccessible(true);
                }
                Element fElt = new Element("field");
                fElt.setAttribute("name", fields[i].getName());
                Class declClass = fields[i].getDeclaringClass();
                fElt.setAttribute("declaringclass", declClass.getName());

                Class fieldType = fields[i].getType();
                Object child = fields[i].get(source);

                if (Modifier.isTransient(fields[i].getModifiers())) {
                    child = null;
                }
                fElt.addContent(serializeVariable(fieldType, child, target, table));
                oElt.addContent(fElt);
            }
        } else {
            Class componentType = sourceClass.getComponentType();

            int length = Array.getLength(source);
            oElt.setAttribute("length", Integer.toString(length));
            for (int i = 0; i < length; i++) {
                oElt.addContent(serializeVariable(componentType,
                        Array.get(source, i),
                        target,
                        table));
            }
        }

        return target;
    }

    private static Element serializeVariable(Class fieldType,
                                             Object child,
                                             Document target,
                                             Map table) throws Exception {
        if (child == null) {
            return new Element("null");
        } else if (!fieldType.isPrimitive()) {
            Element reference = new Element("reference");
            if (table.containsKey(child)) {
                reference.setText(table.get(child).toString());
            } else {
                reference.setText(Integer.toString(table.size()));
                serializeHelper(child, target, table);
            }
            return reference;
        } else {
            Element value = new Element("value");
            value.setText(child.toString());
            return value;
        }
    }
}
