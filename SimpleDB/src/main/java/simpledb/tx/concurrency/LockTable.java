package simpledb.tx.concurrency;

import java.util.*;
import simpledb.file.BlockId;

/**
 * The lock table, which provides methods to lock and unlock blocks.
 * If a transaction requests a lock that causes a conflict with an
 * existing lock, then that transaction is placed on a wait list.
 * There is only one wait list for all blocks.
 * When the last lock on a block is unlocked, then all transactions
 * are removed from the wait list and rescheduled.
 * If one of those transactions discovers that the lock it is waiting for
 * is still locked, it will place itself back on the wait list.
 * @author Edward Sciore
 */
class LockTable {
   private static final long MAX_TIME = 10000; // 10 seconds
   
   private Map<BlockId,List<Integer>> locks = new HashMap<BlockId,List<Integer>>();
   private Map<Integer, Long> txnTimestamps = new HashMap<>();
   /**
    * Grant an SLock on the specified block.
    * If an XLock exists when the method is called,
    * then the calling thread will be placed on a wait list
    * until the lock is released.
    * If the thread remains on the wait list for a certain 
    * amount of time (currently 10 seconds),
    * then an exception is thrown.
    * @param blk a reference to the disk block
    */
   public synchronized void sLock(BlockId blk) throws LockAbortException {
      try {
         long timestamp = System.currentTimeMillis();
         while (hasXlock(blk) && !waitingTooLong(timestamp))
            wait(MAX_TIME);
         if (hasXlock(blk))
            throw new LockAbortException();
         List<Integer> holders = locks.getOrDefault(blk, new ArrayList<>());
         if (!holders.isEmpty()) {
            int oldestTxn = holders.stream().min(Comparator.comparingLong(txnTimestamps::get)).orElseThrow();
            if (txnTimestamps.get(Transaction.current().getId()) < txnTimestamps.get(oldestTxn))
               throw new LockAbortException();
         }
         holders.add(Transaction.current().getId());
         locks.put(blk, holders);
         txnTimestamps.put(Transaction.current().getId(), timestamp);
      } catch(InterruptedException e) {
         throw new LockAbortException();
      }
   }
   
   /**
    * Grant an XLock on the specified block.
    * If a lock of any type exists when the method is called,
    * then the calling thread will be placed on a wait list
    * until the locks are released.
    * If the thread remains on the wait list for a certain 
    * amount of time (currently 10 seconds),
    * then an exception is thrown.
    * @param blk a reference to the disk block
    */
   synchronized void xLock(BlockId blk) throws LockAbortException {
      try {
         long timestamp = System.currentTimeMillis();
         while (hasOtherSLocks(blk) && !waitingTooLong(timestamp))
            wait(MAX_TIME);
         if (hasOtherSLocks(blk))
            throw new LockAbortException();
         List<Integer> holders = locks.getOrDefault(blk, new ArrayList<>());
         if (!holders.isEmpty()) {
            int oldestTxn = holders.stream().min(Comparator.comparingLong(txnTimestamps::get)).orElseThrow();
            if (txnTimestamps.get(Transaction.current().getId()) < txnTimestamps.get(oldestTxn))
               throw new LockAbortException();
         }
         holders.add(-Transaction.current().getId()); // Use negative transaction ID to denote exclusive lock
         locks.put(blk, holders);
         txnTimestamps.put(Transaction.current().getId(), timestamp);
      } catch(InterruptedException e) {
         throw new LockAbortException();
      }
   }
   
   /**
    * Release a lock on the specified block.
    * If this lock is the last lock on that block,
    * then the waiting transactions are notified.
    * @param blk a reference to the disk block
    */
   synchronized void unlock(BlockId blk) {
      List<Integer> holders = locks.getOrDefault(blk, new ArrayList<>());
      holders.remove(Integer.valueOf(Transaction.current().getId()));
      if (holders.isEmpty()) {
         locks.remove(blk);
         notifyAll();
      }
   }
   
   private boolean hasXlock(BlockId blk) {
      List<Integer> holders = locks.getOrDefault(blk, new ArrayList<>());
      return holders.stream().anyMatch(id -> id < 0);
   }
   
   private boolean hasOtherSLocks(BlockId blk) {
      List<Integer> holders = locks.getOrDefault(blk, new ArrayList<>());
      return holders.size() > 1 || (holders.size() == 1 && holders.get(0) < 0);
   }
   
   private boolean waitingTooLong(long starttime) {
      return System.currentTimeMillis() - starttime > MAX_TIME;
   }
   
   private int getLockVal(BlockId blk) {
      Integer ival = locks.get(blk);
      return (ival == null) ? 0 : ival.intValue();
   }
}
