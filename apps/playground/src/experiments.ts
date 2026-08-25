export interface StoredExperiment {
  id: string;
  name: string;
  createdAt: string;
  dataset: string;
  optimizer: string;
  hidden: number;
  learningRate: number;
  metrics: { epoch: number; loss: number; accuracy: number };
  history: Array<{ epoch: number; loss: number; accuracy: number }>;
  model: unknown;
}

const databaseName = "gradlith";
const storeName = "experiments";

export async function saveExperiment(experiment: Omit<StoredExperiment, "id" | "createdAt">): Promise<StoredExperiment> {
  const stored = {
    ...experiment,
    id: crypto.randomUUID(),
    createdAt: new Date().toISOString()
  };
  const db = await openDatabase();
  await requestToPromise(db.transaction(storeName, "readwrite").objectStore(storeName).put(stored));
  db.close();
  return stored;
}

export async function listExperiments(): Promise<StoredExperiment[]> {
  const db = await openDatabase();
  const result = await requestToPromise<StoredExperiment[]>(db.transaction(storeName, "readonly").objectStore(storeName).getAll());
  db.close();
  return result.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function deleteExperiment(id: string): Promise<void> {
  const db = await openDatabase();
  await requestToPromise(db.transaction(storeName, "readwrite").objectStore(storeName).delete(id));
  db.close();
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(storeName, { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function requestToPromise<T = unknown>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

