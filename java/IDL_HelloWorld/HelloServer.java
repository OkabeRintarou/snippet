import HelloApp.*;
import org.omg.CosNaming.*;
import org.omg.CosNaming.NamingContextPackage.*;
import org.omg.CORBA.*;
import org.omg.PortableServer.*;
import org.omg.PortableServer.POA;

import java.util.Properties;

class HelloImpl extends HelloPOA {
	private ORB orb;

	public void setORB(ORB orb_val) {
		orb = orb_val;
	}

	public String sayHello() {
		return "\nHello world !!\n";
	}

	public void shutdown() {
		orb.shutdown(false);
	}
}

public class HelloServer {
	public static void main(String[] args) {
		try {
			// A CORBA server needs a local ORB object, as does the CORBA client
			ORB orb = ORB.init(args,null);
			POA rootpoa = POAHelper.narrow(orb.resolve_initial_references("RootPOA"));
			rootpoa.the_POAManager().activate();

			HelloImpl helloImpl = new HelloImpl();
			helloImpl.setORB(orb);

			org.omg.CORBA.Object ref = rootpoa.servant_to_reference(helloImpl);
			Hello href = HelloHelper.narrow(ref);

			org.omg.CORBA.Object objRef = 
				orb.resolve_initial_references("NameService");

			NamingContextExt ncRef = NamingContextExtHelper.narrow(objRef);
			String name = "Hello";
			NameComponent path[] = ncRef.to_name(name);
			ncRef.rebind(path,href);

			orb.run();
		} catch (Exception e) {
			System.err.println("ERROR: " + e);
			e.printStackTrace(System.out);
		}

		System.out.println("HelloServer Exiting...");
	}
}


