package ch04;

import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Callable;

public class AlarmAgent implements Runnable {
    // 用于记录AlarmAgent是否连上告警服务器
    private volatile boolean connectedToServer = false;

    // 模式角色: GuardedSuspension.Predicate
    private final Predicate agentConnected = new Predicate() {
        @Override
        public boolean evaluate() {
            return connectedToServer;
        }
    };

    // 模式角色: GuardedSuspension.Blocker
    private final Blocker blocker = new ConditionVarBlocker();

    // 心跳计时器
    private final Timer heartbeatTimer = new Timer(true);

        public void sendAlarm(final AlarmInfo alarm) throws Exception {
            GuardedAction<Void> guardedAction = new GuardedAction<Void>(agentConnected) {
                @Override
                public Void call() throws Exception {
                    doSendAlarm(alarm);
                    return null;
                }
            };

        blocker.callWithGuard(guardedAction);
    }

    // 通过网络连接将告警信息发送给告警服务器
    private void doSendAlarm(AlarmInfo alarm) {
        try {
            Thread.sleep(50);
            System.out.println("send alarm information");
        } catch(Exception e) {

        }
    }

    private void init() {
        Thread connectingThread = new Thread(new ConnectingTask());
        connectingThread.start();
        heartbeatTimer.schedule(new HeartbeatTask(), 60000,2000);
    }

    @Override
    public void run(){
        init();

        // collect alarm information

        for (;;) {
            AlarmInfo alarm = new AlarmInfo();
            try {
                Thread.sleep(1000);
                sendAlarm(alarm);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public void disconnect() {
        System.out.println("disconnected from alarm server");
        connectedToServer = false;
    }

    protected void onConnected() {
        try {
            blocker.signalAfter(new Callable<Boolean>() {
                @Override
                public Boolean call() throws Exception {
                    connectedToServer = true;
                    System.out.println("connect to server");
                    return Boolean.TRUE;
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected void onDisconnected() {
        connectedToServer = false;
    }

    private class ConnectingTask implements Runnable {
        @Override
        public void run() {
            // 模拟连接操作耗时
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            onConnected();
        }
    }

    private class HeartbeatTask extends TimerTask {

        @Override
        public void run() {
            if (!testConnection()) {
                onDisconnected();
                reconnect();
            }
        }

        private boolean testConnection() {
            return true;
        }

        private void reconnect() {
            ConnectingTask connectingThread = new ConnectingTask();

            //直接在心跳计时器线程中执行
            connectingThread.run();
        }
    }

    public static void main(String[] args) {
        new AlarmAgent().run();
    }
}
