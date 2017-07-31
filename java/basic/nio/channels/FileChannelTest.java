import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;

public class FileChannelTest {
	public static void main(String[] args)throws Exception{
		RandomAccessFile aFile = new RandomAccessFile("FileChannelTest.java","r");
		FileChannel fileChannel = aFile.getChannel();
		ByteBuffer byteBuffer = ByteBuffer.allocate(48);
		int bytesRead = fileChannel.read(byteBuffer);
		while(bytesRead != -1){
			System.out.println("[Read] " + bytesRead);
			byteBuffer.flip();
			while(byteBuffer.hasRemaining()){
				System.out.println((char)byteBuffer.get());
			}
			byteBuffer.clear();
			bytesRead = fileChannel.read(byteBuffer);
		}
	}
}
