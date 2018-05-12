/*
 * VM Args: -Xss2M
 */
public class JavaVMStackOOM {
  private void dontStop() {
    for (;;) {
    }
  }

  public void stackLeakByThread() {
    for (;;) {
      Thread thread = new Thread(new Runnable() {
        @Override
        public void run() {
          dontStop();
        }
      });
    }
  }

  public static void main(String[] args) {
    JavaVMStackOOM oom = new JavaVMStackOOM();
    oom.stackLeakByThread();
  }
}
