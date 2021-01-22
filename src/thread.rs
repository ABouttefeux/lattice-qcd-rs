
use  std::{
    collections::HashMap,
    iter::Iterator,
    sync::{
        Arc,
        Mutex,
        mpsc,
    },
    any::Any,
    hash::Hash,
};
use crossbeam::{
    //channel,
    thread,
};

#[derive(Debug)]
pub enum ThreadError {
    ThreadNumberIncorect,
    Panic(Box<dyn Any + Send + 'static>),
}


pub fn run_pool_parallel<Key, Data, CommonData, F>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    number_of_thread: usize,
    capacity: usize,
) -> Result<HashMap<Key, Data>, ThreadError>
    where //Builder: BuildHasher + Default,
    CommonData: Sync,
    Key: Eq + Hash + Send + Clone + Sync,
    Data: Send,
    F: Fn(&Key, &CommonData) -> Data,
    F: Sync + Clone,
{
    run_pool_parallel_with_initialisation_mutable(
        iter,
        common_data,
        &|_, key, common| { closure(key, common)},
        &|| (),
        number_of_thread,
        capacity,
    )
}

pub fn run_pool_parallel_with_initialisation_mutable<Key, Data, CommonData, InitData, F, FInit>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    closure_init: FInit,
    number_of_thread: usize,
    capacity: usize,
) -> Result<HashMap<Key, Data>, ThreadError>
    where //Builder: BuildHasher + Default,
    CommonData: Sync,
    Key: Eq + Hash + Send + Clone + Sync,
    Data: Send,
    F: Fn(&mut InitData, &Key, &CommonData) -> Data,
    F: Sync + Clone,
    FInit: FnOnce() -> InitData,
    FInit: Send + Clone,
{
    if number_of_thread == 0 {
        return Err(ThreadError::ThreadNumberIncorect);
    }
    else if number_of_thread== 1{
        let mut hash_map = HashMap::<Key, Data>::with_capacity(capacity);
        let mut init_data = closure_init();
        for i in iter{
            hash_map.insert(i.clone(), closure(&mut init_data, &i, common_data));
        }
        return Ok(hash_map);
    }
    else {
        let result = thread::scope(|s| {
            let mutex_iter = Arc::new(Mutex::new(iter));
            let mut threads = vec![];
            let (resut_tx, resut_rx) = mpsc::channel::<(Key, Data)>();
            for _ in 0..number_of_thread {
                let iter_clone = Arc::clone(&mutex_iter);
                let transmitter = resut_tx.clone();
                let closure_init_clone = closure_init.clone();
                let handel = s.spawn(move |_| {
                    let mut init_data = closure_init_clone();
                    loop {
                        let val = iter_clone.lock().unwrap().next();
                        match val {
                            Some(i) => transmitter.send((i.clone(), closure(&mut init_data, &i, common_data))).unwrap(),
                            None => break,
                        }
                    }
                });
                threads.push(handel);
            }
            // we drop channel so we can proprely assert if they are closed
            drop(resut_tx);
            let mut hash_map = HashMap::<Key, Data>::with_capacity(capacity);
            for message in resut_rx {
                let (key, data) = message;
                hash_map.insert(key, data);
            }
            for handel in threads {
                handel.join().map_err(|err| ThreadError::Panic(err) )?;
            }
            return Ok(hash_map);
        }).map_err(|err| ThreadError::Panic(err))?;
        return result;
    }
}
