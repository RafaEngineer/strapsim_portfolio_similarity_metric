<!DOCTYPE html>
<html lang="pt-br">
<head>
    <title>Sistema Crud</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="crud">
   <h3>Cadastro de Usuário</h3>
    <form method="post" action="">
        <input name="nome" type="text" placeholder="Seu nome">
        <input name="email" type="email" placeholder="Seu e-mail">
        <textarea name="descricao" placeholder="Descreva-se"></textarea>
        <button type="submit">Cadastrar</button>
    </form>
    <?php
     require('../PDWEL-master/Database.php');
     $nome = filter_input(INPUT_POST, 'nome', FILTER_SANITIZE_STRING);
     $email = filter_input(INPUT_POST, 'email', FILTER_VALIDATE_EMAIL);
     $descricao = filter_input(INPUT_POST, 'descricao', FILTER_SANITIZE_SPECIAL_CHARS);
     if(!empty($nome) && !empty($email) && !empty($descricao)){
         $DataBase = new Database();
         $sql = "INSERT INTO usuarios SET nome = :nome, email = :email, descricao = :descricao";
         $binds = ['nome' => $nome, 'email' => $email, 'descricao' => $descricao];
         $result = $DataBase->insert($sql, $binds);
         if($result){
             echo "<div class='success'>Cadastro foi um sucesso</div>";
         }else{
             echo "Ops, houve um erro no cadastro";
         }
     }



    ?>
</div>
</body>
</html>