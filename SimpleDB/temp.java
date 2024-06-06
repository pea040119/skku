package simpledb.buffer;

import simpledb.file.*;
import simpledb.log.LogMgr;

import java.util.*;

public class BufferMgr {
   private Map<BlockId, Buffer> bufferMap; // 할당된 버퍼를 관리하는 맵
   private LinkedList<Buffer> unpinnedBuffers; // LRU 순서로 unpinned 버퍼를 관리하는 리스트
   private int numAvailable;
   private static final long MAX_TIME = 10000; // 10 seconds
   private FileMgr fm;
   private LogMgr lm;

   /**
    * Constructor:  Creates a buffer manager having the specified
    * number of buffer slots.
    * This constructor depends on a {@link FileMgr} and
    * {@link simpledb.log.LogMgr LogMgr} object.
    * @param numbuffs the number of buffer slots to allocate
    */
   public BufferMgr(FileMgr fm, LogMgr lm, int numbuffs) {
      this.fm = fm;
      this.lm = lm;
      bufferMap = new HashMap<>();
      unpinnedBuffers = new LinkedList<>();
      numAvailable = numbuffs;
      for (int i = 0; i < numbuffs; i++) {
         Buffer buff = new Buffer(fm, lm, i);
         unpinnedBuffers.add(buff);
      }
   }

   /**
    * Returns the number of available (i.e. unpinned) buffers.
    * @return the number of available buffers
    */
   public synchronized int available() {
      return numAvailable;
   }

   /**
    * Flushes the dirty buffers modified by the specified transaction.
    * @param txnum the transaction's id number
    */
   public synchronized void flushAll(int txnum) {
      for (Buffer buff : bufferMap.values()) {
         if (buff.modifyingTx() == txnum)
            buff.flush();
      }
   }

   /**
    * Unpins the specified data buffer. If its pin count
    * goes to zero, then notify any waiting threads.
    * @param buff the buffer to be unpinned
    */
   public synchronized void unpin(Buffer buff) {
      buff.unpin();
      if (!buff.isPinned()) {
         numAvailable++;
         unpinnedBuffers.add(buff);
         notifyAll();
      }
   }

   /**
    * Pins a buffer to the specified block, potentially
    * waiting until a buffer becomes available.
    * If no buffer becomes available within a fixed
    * time period, then a {@link BufferAbortException} is thrown.
    * @param blk a reference to a disk block
    * @return the buffer pinned to that block
    */
   public synchronized Buffer pin(BlockId blk) {
      try {
         long timestamp = System.currentTimeMillis();
         Buffer buff = tryToPin(blk);
         while (buff == null && !waitingTooLong(timestamp)) {
            wait(MAX_TIME);
            buff = tryToPin(blk);
         }
         if (buff == null)
            throw new BufferAbortException();
         return buff;
      } catch (InterruptedException e) {
         throw new BufferAbortException();
      }
   }

   /**
    * Returns true if starttime is older than 10 seconds
    * @param starttime timestamp
    * @return true if waited for more than 10 seconds
    */
   private boolean waitingTooLong(long starttime) {
      return System.currentTimeMillis() - starttime > MAX_TIME;
   }

   /**
    * Tries to pin a buffer to the specified block.
    * If there is already a buffer assigned to that block
    * then that buffer is used;
    * otherwise, an unpinned buffer from the pool is chosen.
    * Returns a null value if there are no available buffers.
    * @param blk a reference to a disk block
    * @return the pinned buffer
    */
   private Buffer tryToPin(BlockId blk) {
      Buffer buff = bufferMap.get(blk);
      if (buff == null) {
         buff = chooseUnpinnedBuffer();
         if (buff == null)
            return null;
         buff.assignToBlock(blk);
         bufferMap.put(blk, buff);
      }
      if (!buff.isPinned())
         numAvailable--;
      buff.pin();
      return buff;
   }

   /**
    * Find and return an unpinned buffer.
    * @return the unpinned buffer
    */
   private Buffer chooseUnpinnedBuffer() {
      if (unpinnedBuffers.isEmpty())
         return null;
      return unpinnedBuffers.removeFirst();
   }

   /**
    * Print the current status of the buffer manager.
    */
   public synchronized void printStatus() {
      System.out.println("Allocated Buffers:");
      for (Buffer buff : bufferMap.values()) {
         String status = buff.isPinned() ? "pinned" : "unpinned";
         System.out.println("Buffer " + buff.getId() + ": " + buff.block() + " " + status);
      }
      System.out.print("Unpinned Buffers in LRU order: ");
      for (Buffer buff : unpinnedBuffers) {
         System.out.print(buff.getId() + " ");
      }
      System.out.println();
   }
}
