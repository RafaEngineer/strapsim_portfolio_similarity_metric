<?php

class database{
	private $PDO;
	

	public function __construct($dbname = 'cadastro')
	{
		//$dsn = 'mysql:dbname={$dbname},host=localhost';
		//$user = 'root';
		//$password = '';
		try{
			//$this->PDO = NEW PDO($dsn, $user, $password);
			$this->PDO = new PDO("mysql:host=127.0.0.1;dbname={$dbname}",'root','');
			//$this ->PDO->setAttribute( attribute PDO::ATTR_ERRMODE, value: PDO::ERRMODE_EXCEPTION);
		}catch (PDOException $e){
			die("Ops, houve um erro: <b> {$e ->	getMessage()} </b>");
		}
	}

	public function insert($sql, array $binds){
		$stmt = $this->PDO->prepare($sql);

		foreach ($binds as $key => $value){
			$stmt -> bindValue($key, $value);
		}
	
	$stmt->execute();
	if($stmt->rowCount() > 0){
		return true;
	}
	return false;
}

	public function select($sql, array $binds){
		$stmt = $this->PDO->prepare($sql);
		foreach($binds as $key => $value){
			$stmt->bindValue($key, $value);
		}
		$stmt->execute();
		return $stmt;
	}

	public function update($sql, array $binds){
		$stmt = $this->PDO->prepare($sql);
		foreach($binds as $key =>$value){
			$stmt->bindValue($key, $value);
		}
		$stmt->execute();
		return $stmt->rowCount();
	}
	public function delete($sql, array $binds){
		$stmt = $this->PDO->prepare($sql);
		foreach($binds as $key=> $value){
			$stmt->bindValue($key, $value);
		}	
	$stmt->execute();
	return $stmt->rowCount();
	}
}