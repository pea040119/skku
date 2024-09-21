#include <iostream>
#include <sstream>
#include <pthread.h>
#include <queue>

#define BILLION  1000000000L

using namespace std;
 


//SkipList Node
template<class K,class V,int MAXLEVEL>
class skiplist_node {
public:
    skiplist_node() {
        for ( int i=1; i<=MAXLEVEL; i++ ) {
           forwards[i] = NULL;
        }
        pthread_rwlock_init(&lock, NULL);
    }
 

    skiplist_node(K searchKey):key(searchKey) {
        for ( int i=1; i<=MAXLEVEL; i++ ) {
            forwards[i] = NULL;
        }
        pthread_rwlock_init(&lock, NULL);
    }
 

    skiplist_node(K searchKey,V val):key(searchKey),value(val) {
        for ( int i=1; i<=MAXLEVEL; i++ ) {
            forwards[i] = NULL;
        }
        pthread_rwlock_init(&lock, NULL);
    }
 
    virtual ~skiplist_node() {
        pthread_rwlock_destroy(&lock);
    }


    void read_lock() {
        pthread_rwlock_rdlock(&lock);
    }
    

    void write_lock() {
        pthread_rwlock_wrlock(&lock);
    }

    
    void unlock() {
        pthread_rwlock_unlock(&lock);
    }


    K key;
    V value;
    skiplist_node<K,V,MAXLEVEL>* forwards[MAXLEVEL+1];


private:
    pthread_rwlock_t lock;
};
 


//SkipList
template<class K, class V, int MAXLEVEL = 16>
class skiplist
{///////////////////////////////////////////////////////////////////////////////
public:
    const int max_level;
    typedef K KeyType;
    typedef V ValueType;
    typedef skiplist_node<K,V,MAXLEVEL> NodeType;
 

    skiplist(K minKey,K maxKey):m_pHeader(NULL),m_pTail(NULL),
                                max_curr_level(1),max_level(MAXLEVEL),
                                m_minKey(minKey),m_maxKey(maxKey)
    {
        m_pHeader = new NodeType(m_minKey);
        m_pTail = new NodeType(m_maxKey);
        for ( int i=1; i<=MAconst int max_level;XLEVEL; i++ ) {
            m_pHeader->forwards[i] = m_pTail;
        }
    }
 

    virtual ~skiplist()
    {
        NodeType* currNode = m_pHeader->forwards[1];
        while ( currNode != m_pTail ) {
            NodeType* tempNode = currNode;
            currNode = currNode->forwards[1];
            delete tempNode;
        }
        delete m_pHeader;
        delete m_pTail;
    }


    void insert(K searchKey, V newValue) {

    }


    V find(K searchKey) {

    }


    bool empty() const
    {
        return ( m_pHeader->forwards[1] == m_pTail );
    }


private:
    queue<int, int> buffer;
    pthread_mutex_t buffer_lock;
    pthread_cond_t producer_cond;
    pthread_cond_t consumer_cond;


    void _skiplist() {

    }


    void _insert(K searchKey,V newValue)
    {
        skiplist_node<K,V,MAXLEVEL>* update[MAXLEVEL];
        NodeType *currNode = m_pHeader, *temp;
        bool check = false;
        
        for(int level=max_curr_level; level >=1; level--) {
            while (true) {
                currNode->read_lock();
                if (currNode->forwards[level]->key >= searchKey) {
                    currNode->unlock();
                    break;
                }
                temp = currNode->forwards[level];
                currNode->unlock();
                currNode = temp;
            }
            update[level] = currNode;
        }
        currNode->read_lock();
        temp = currNode->forwards[1];
        currNode->unlock();
        currNode = temp;

        currNode->read_lock();
        check = currNode->key == searchKey;
        currNode->unlock();

        
        if (check) {
            currNode->write_lock();
            currNode->value = newValue;
            currNode->unlock();
        }
        else {
            int newlevel = randomLevel();
            if ( newlevel > max_curr_level ) {
                for ( int level = max_curr_level+1; level <= newlevel; level++ ) {
                    update[level] = m_pHeader;
                }
                max_curr_level = newlevel;
            }
            currNode = new NodeType(searchKey,newValue);
            currNode->write_lock();
            for ( int lv=1; lv<=max_curr_level; lv++ ) {
                update[lv]->write_lock();
                currNode->forwards[lv] = update[lv]->forwards[lv];
                update[lv]->forwards[lv] = currNode;
                update[lv]->unlock();
            }
            currNode->unlock();
        }
    }
 
    void erase(K searchKey)
    {
        skiplist_node<K,V,MAXLEVEL>* update[MAXLEVEL];
        NodeType* currNode = m_pHeader;
        for(int level=max_curr_level; level >=1; level--) {
            while ( currNode->forwards[level]->key < searchKey ) {
                currNode = currNode->forwards[level];
            }
            update[level] = currNode;
        }
        currNode = currNode->forwards[1];
        if ( currNode->key == searchKey ) {
            for ( int lv = 1; lv <= max_curr_level; lv++ ) {
                if ( update[lv]->forwards[lv] != currNode ) {
                    break;
                }
                update[lv]->forwards[lv] = currNode->forwards[lv];
            }
            delete currNode;
            // update the max level
            while ( max_curr_level > 1 && m_pHeader->forwards[max_curr_level] == NULL ) {
                max_curr_level--;
            }
        }
    }
 
    //const NodeType* find(K searchKey)
    V _find(K searchKey) {
        NodeType *currNode = m_pHeader, *temp;
        bool check;

        for(int level=max_curr_level; level >=1; level--) {
            while (true) {
                currNode->read_lock();
                if (currNode->forwards[level]->key >= searchKey) {
                    currNode->unlock();
                    break;
                }
                temp = currNode->forwards[level];
                currNode->unlock();
                currNode = temp;
            }
        }

        currNode->read_lock();
        temp = currNode->forwards[1];
        currNode->unlock();
        currNode = temp;
        currNode->read_lock();
        check = currNode->key == searchKey;
        currNode->unlock();
        if (check) {
            return currNode->value;
        }
        else {
            //return NULL;
            return -1;
        }
    }
 
 
    std::string _printList()
    {
	int i=0;
        std::stringstream sstr;
        NodeType* currNode = m_pHeader->forwards[1];
        while ( currNode != m_pTail ) {
            //sstr << "(" << currNode->key << "," << currNode->value << ")" << endl;
            sstr << currNode->key << " ";
            currNode = currNode->forwards[1];
	    i++;
	    if(i>200) break;
        }
        return sstr.str();
    }


 
protected:
    double uniformRandom()
    {
        return rand() / double(RAND_MAX);
    }
 
    int randomLevel() {
        int level = 1;
        double p = 0.5;
        while ( uniformRandom() < p && level < MAXLEVEL ) {
            level++;
        }
        return level;
    }
    K m_minKey;
    K m_maxKey;
    int max_curr_level;
    skiplist_node<K,V,MAXLEVEL>* m_pHeader;
    skiplist_node<K,V,MAXLEVEL>* m_pTail;
};
 
///////////////////////////////////////////////////////////////////////////////
 
