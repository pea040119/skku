#include <iostream>
#include <sstream>

#define BILLION  1000000000L

using namespace std;
 
// SkipList Node
template<class K,class V,int MAXLEVEL>
class skiplist_node {
private:
    pthread_rwlock_t lock;


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
};

///////////////////////////////////////////////////////////////////////////////
 
template<class K, class V, int MAXLEVEL = 16>
class skiplist
{
public:
    typedef K KeyType;
    typedef V ValueType;
    typedef skiplist_node<K,V,MAXLEVEL> NodeType;
 
    skiplist(K minKey,K maxKey):m_pHeader(NULL),m_pTail(NULL),
                                max_curr_level(1),max_level(MAXLEVEL),
                                m_minKey(minKey),m_maxKey(maxKey)
    {
        m_pHeader = new NodeType(m_minKey);
        m_pTail = new NodeType(m_maxKey);
        for ( int i=1; i<=MAXLEVEL; i++ ) {
            m_pHeader->forwards[i] = m_pTail;
        }
    }
 
    virtual ~skiplist()
    {
        NodeType* curr_node = m_pHeader->forwards[1];
        while ( curr_node != m_pTail ) {
            NodeType* tempNode = curr_node;
            curr_node = curr_node->forwards[1];
            delete tempNode;
        }
        delete m_pHeader;
        delete m_pTail;
    }
 
    void insert(K searchKey,V newValue)
    {
        skiplist_node<K,V,MAXLEVEL>* update[MAXLEVEL];
        NodeType *curr_node = m_pHeader, *temp;
        bool check = false;
        
        for(int level=max_curr_level; level >=1; level--) {
            while (true) {
                curr_node->read_lock();
                if (curr_node->forwards[level]->key >= searchKey) {
                    curr_node->unlock();
                    break;
                }
                temp = curr_node->forwards[level];
                curr_node->unlock();
                curr_node = temp;
            }
            update[level] = curr_node;
        }
        curr_node->read_lock();
        temp = curr_node->forwards[1];
        curr_node->unlock();
        curr_node = temp;

        curr_node->read_lock();
        check = curr_node->key == searchKey;
        curr_node->unlock();

        if (check) {
            curr_node->write_lock();
            curr_node->value = newValue;
            curr_node->unlock();
        }
        else {
            int newlevel = randomLevel();
            if ( newlevel > max_curr_level ) {
                for ( int level = max_curr_level+1; level <= newlevel; level++ ) {
                    update[level] = m_pHeader;
                }
                max_curr_level = newlevel;
            }
            curr_node = new NodeType(searchKey,newValue);
            curr_node->write_lock();
            for ( int lv=1; lv<=max_curr_level; lv++ ) {
                update[lv]->write_lock();
                curr_node->forwards[lv] = update[lv]->forwards[lv];
                update[lv]->forwards[lv] = curr_node;
                update[lv]->unlock();
            }
            curr_node->unlock();
        }
    }
 
    void erase(K searchKey)
    {
        skiplist_node<K,V,MAXLEVEL>* update[MAXLEVEL];
        NodeType* curr_node = m_pHeader;
        for(int level=max_curr_level; level >=1; level--) {
            while ( curr_node->forwards[level]->key < searchKey ) {
                curr_node = curr_node->forwards[level];
            }
            update[level] = curr_node;
        }
        curr_node = curr_node->forwards[1];
        if ( curr_node->key == searchKey ) {
            for ( int lv = 1; lv <= max_curr_level; lv++ ) {
                if ( update[lv]->forwards[lv] != curr_node ) {
                    break;
                }
                update[lv]->forwards[lv] = curr_node->forwards[lv];
            }
            delete curr_node;
            // update the max level
            while ( max_curr_level > 1 && m_pHeader->forwards[max_curr_level] == NULL ) {
                max_curr_level--;
            }
        }
    }
 
    //const NodeType* find(K searchKey)
    V find(K searchKey)
    {
        NodeType *curr_node = m_pHeader, *temp;
        bool check;

        for(int level=max_curr_level; level >=1; level--) {
            while (true) {
                curr_node->read_lock();
                if (curr_node->forwards[level]->key >= searchKey) {
                    curr_node->unlock();
                    break;
                }
                temp = curr_node->forwards[level];
                curr_node->unlock();
                curr_node = temp;
            }
        }

        curr_node->read_lock();
        temp = curr_node->forwards[1];
        curr_node->unlock();
        curr_node = temp;
        curr_node->read_lock();
        check = curr_node->key == searchKey;
        curr_node->unlock();
        V value = curr_node->value;
        if (check) {
            return value;
        }
        else
            return -1;
    }
 
    bool empty() const
    {
        return ( m_pHeader->forwards[1] == m_pTail );
    }
 
    std::string printList()
    {
	int i=0;
        std::stringstream sstr;
        NodeType* curr_node = m_pHeader->forwards[1];
        while ( curr_node != m_pTail ) {
            //sstr << "(" << curr_node->key << "," << curr_node->value << ")" << endl;
            sstr << curr_node->key << " ";
            curr_node = curr_node->forwards[1];
	    i++;
	    if(i>200) break;
        }
        return sstr.str();
    }
 
    const int max_level;
 
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
 
