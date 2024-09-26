// SkipList v0.1

#include <iostream>
#include <sstream>
#include <pthread.h>

#define BILLION  1000000000L

using namespace std;
 


// Linked List Node
template<class K,class V>
class linked_list_node {
public:
    linked_list_node* next;
    int action;
    K key;
    V value;


    linked_list_node() {
        next = NULL;
        action = -1;
        value = -1;
        key = -1;
        pthread_rwlock_init(&lock, NULL);
    }


    virtual ~linked_list_node() {
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


private:
    pthread_rwlock_t lock;
};



// Circular Linked List
template <class K, class V>
class circular_linked_list {
private:
    typedef linked_list_node<K, V> NodeType;
    int _size;
    NodeType *head, *tail;


public:
    circular_linked_list(int size) {
        _size = size;

        head = new NodeType();
        NodeType *curr_node = head, *temp;
        for(int i=0; i<size; i++) {
            curr_node->write_lock();
            curr_node->next = new NodeType();
            temp = curr_node->next;
            curr_node->unlock();
            curr_node = temp;
        }
        curr_node->write_lock();
        curr_node->next = head;
        curr_node->unlock();
        tail = head;
    }


    ~circular_linked_list() {
        NodeType *curr_node = head;
        for (int i = 0; i < _size; i++) {
            curr_node->write_lock();
            NodeType *temp = curr_node->next;
            curr_node->unlock();
            delete curr_node;
            curr_node = temp;
        }
    }


    bool push(int action, K key, V value) {
        NodeType *temp;
        tail->write_lock();

        if (tail->next == head) {
            tail->unlock();
            return false;
        }

        tail->action = action;
        tail->key = key;
        tail->value = value;
        temp = tail->next;

        tail->unlock();
        tail = temp;

        return true;   
    }


    bool pop(int *action, K* key, V *value) {
        NodeType *temp;
        head->write_lock();
        
        if (head->action == -1) {
            head->unlock();
            return false;
        }

        *action = head->action;
        *key = head->key;
        *value = head->value;
        head->action = -1;
        temp = head->next;

        head->unlock();
        head = temp;
        return true;
    }


    int size() {
        return _size;
    }


    bool is_empty() {
        bool check;
        head->read_lock();
        check = head->next == tail;
        head->unlock();
        return check;
    }


    bool is_full() {
        bool check;
        tail->read_lock();
        check = tail->next == head;
        tail->unlock();
        return check;
    }


    void print() {
        NodeType *curr_node = head, *temp;
        cout << "Linked List"<<endl;
        while(true) {
            curr_node->read_lock();
            if (curr_node->action == -1) {
                curr_node->unlock();
                break;
            }
            cout << "(action: " << curr_node->action << "\tkey: " << curr_node->key << "\tvalue: " << curr_node->value << ")" << endl;
            temp = curr_node->next;
            curr_node->unlock();
            curr_node = temp;
        }
        cout << "end" << endl;
    }
};



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
 


//SkipList
template<class K, class V, int MAXLEVEL = 16>
class skiplist{
private:
    const int max_level;
    typedef K KeyType;
    typedef V ValueType;
    typedef skiplist_node<K,V,MAXLEVEL> NodeType;

    int num_thread;
    pthread_t *threads;
    int num_run_thread;
    int num_insert_thread;
    bool pause;
    bool run;

    pthread_mutex_t mutex;
    pthread_cond_t cond;


    static void* proc(void* arg) {
        skiplist<int, int>* pSelf = static_cast<skiplist<int, int>*>(arg);

        int action;
        K key;
        V value;
        while(pSelf->run) {
            while(pSelf->run) {
                bool check = pSelf->input_buffer->pop(&action, &key, &value);
                if (check) {
                    pSelf->num_run_thread +=1;
                    break;
                }
            }
            switch(action) {
            case 0:
                pSelf->_insert(key, value);
                break;
            case 1:

                pSelf->_find(key);
            }
            pSelf->num_run_thread -=1;
        }

        return nullptr;
    }


    void _insert(K searchKey,V newValue)
    {
        cout << "_insert" << endl;
        skiplist_node<K,V,MAXLEVEL>* update[MAXLEVEL];
        NodeType *curr_node = m_pHeader, *temp;
        bool check = false;
        
        for(int level=max_curr_level; level >=1; level--) {
            while (run) {
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
    void _find(K searchKey) {
        NodeType *curr_node = m_pHeader, *temp;
        bool check;

        for(int level=max_curr_level; level >=1; level--) {
            while (run) {
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
        cout << check << searchKey << endl;
        if (check) {
            curr_node->read_lock();
            sstr << curr_node->value << "\t";
            temp = curr_node->forwards[1];
            curr_node->unlock();
            curr_node = temp;
            curr_node->read_lock();
            sstr << curr_node->value << endl;
            curr_node->unlock();

        }
        else
            sstr << "ERROR: Not Found: " << searchKey << endl;
    }


public:
    circular_linked_list<K, V> *input_buffer;
    std::stringstream sstr;


    skiplist(K minKey,K maxKey, int num_thread = 2):m_pHeader(NULL),m_pTail(NULL),
                                max_curr_level(1),max_level(MAXLEVEL),
                                m_minKey(minKey),m_maxKey(maxKey),num_thread(num_thread) {
        m_pHeader = new NodeType(m_minKey);
        m_pTail = new NodeType(m_maxKey);

        cond = PTHREAD_COND_INITIALIZER;
        mutex = PTHREAD_MUTEX_INITIALIZER;

        for ( int i=1; i<=MAXLEVEL; i++ ) {
            m_pHeader->forwards[i] = m_pTail;
        }
        
        input_buffer = new circular_linked_list<K, V> (100);
        pause = false;
        run = true;

        threads = new pthread_t[num_thread];

        for(int i=0; i<num_thread; i++) {
            pthread_create(&threads[i], NULL, proc, this);
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
        delete input_buffer;
        delete threads;
        delete m_pHeader;
        delete m_pTail;
    }


    void query(int action, long num) {
        input_buffer->push(action, num, num);
        input_buffer->print();
    }

    
    void wait() {
        while(num_run_thread!=0 && !input_buffer->is_empty());
        cout << "end" << endl;
        run = false;
        for(int i=0; i<num_thread; i++)
            pthread_cancel(threads[i]);
    }


    void printList() {
        pause = true;
        while (num_run_thread != 0);

        int i=0;

        m_pHeader->read_lock();
        NodeType* curr_node = m_pHeader->forwards[1], *temp;
        m_pHeader->unlock();

        while ( curr_node != m_pTail ) {
            //sstr << "(" << curr_node->key << "," << curr_node->value << ")" << endl;
            curr_node->read_lock();
            sstr << curr_node->key << " ";
            temp = curr_node->forwards[1];
            curr_node->unlock();
            i++;
            if(i>200) break;
        }
        pause = false;
    }


    bool empty() const {
        bool check;
        m_pHeader->read_lock();
        check = m_pHeader->forwards[1] == m_pTail;
        m_pHeader->unlock();
        return check;
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